import * as tf from "@tensorflow/tfjs";

export let batchedMultiheadedDotProduct = tf.customGrad((query, key, save) => {
    save([query, key]);
    return {
        value: tf.einsum("bihj,bkhj->bihk", query, key), gradFunc: (dy, saved) => {
            return tf.tidy(() => {
                let [q, k] = saved;
                let dfdq = tf.einsum("il,bkhm->bihklm", tf.eye(q.shape[1]), k);
                let dfdk = tf.einsum("bihm,kl->bihklm", q, tf.eye(k.shape[1]));
                if (dy) {
                    return [tf.einsum("bihklm,bihk->blhm", dfdq, dy), tf.einsum("bihklm,bihk->blhm", dfdk, dy)];
                } else {
                    return [dfdq, dfdk];
                }
            })
        }
    }
});

export let batchedMultiheadedMatmul = tf.customGrad((a, b, save) => {
    save([a, b]);
    return {
        value: tf.einsum("bihj,bjhk->bihk", a, b), gradFunc: (dy, saved) => {
            return tf.tidy(() => {
                let [q, k] = saved;
                let dfdq = tf.einsum("il,bmhk->bihklm", tf.eye(q.shape[1]), k);
                let dfdk = tf.einsum("bihl,km->bihklm", q, tf.eye(k.shape[3]));
                if (dy) {
                    return [tf.einsum("bihklm,bihk->blhm", dfdq, dy), tf.einsum("bihklm,bihk->blhm", dfdk, dy)];
                } else {
                    return [dfdq, dfdk];
                }
            })
        }
    }
});