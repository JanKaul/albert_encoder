import * as tf from "@tensorflow/tfjs";
import { nameScopeWrapper } from "./nameScope";
export class AlbertEmbedding extends tf.layers.Layer {
    static className = 'AlbertEmbedding';
    vocabSize: number;
    maxPositions: number;
    typeVocabSize: number;
    embeddingSize: number;
    hiddenDropoutRate: number;
    initializerRange: number;
    epsilon: number;
    kwargs: {};
    layernorm: any;
    dropout: any;
    wordEmbedding: tf.LayerVariable;
    positionEmbedding: tf.LayerVariable;
    segmentEmbedding: tf.LayerVariable;
    /* Embeddings for Albert. */
    constructor({ vocabSize = 30000, maxPositions = 512, typeVocabSize = 2, embeddingSize = 128, hiddenDropoutRate = 0, initializerRange = 0.02, epsilon = 1e-08, name = "embeddings", ...kwargs } = {}) {
        super({ name: name, ...kwargs });
        this.vocabSize = vocabSize;
        this.maxPositions = maxPositions;
        this.typeVocabSize = typeVocabSize;
        this.embeddingSize = embeddingSize;
        this.hiddenDropoutRate = hiddenDropoutRate;
        this.initializerRange = initializerRange;
        this.epsilon = epsilon;
        this.kwargs = kwargs;
        this.layernorm = tf.layers.layerNormalization({ epsilon: epsilon, name: "LayerNorm" });
        this.dropout = tf.layers.dropout({ rate: hiddenDropoutRate });
    }
    build([idShape, segmentShape = undefined]) {
        this.wordEmbedding = this.addWeight("word_embeddings", [this.vocabSize, this.embeddingSize], "float32", tf.initializers.truncatedNormal({ stddev: this.initializerRange }));
        this.positionEmbedding = this.addWeight("position_embeddings", [this.maxPositions, this.embeddingSize], "float32", tf.initializers.truncatedNormal({ stddev: this.initializerRange }));
        this.segmentEmbedding = this.addWeight("token_type_embeddings", [this.typeVocabSize, this.embeddingSize], "float32", tf.initializers.truncatedNormal({ stddev: this.initializerRange }));
        nameScopeWrapper(this.layernorm.name, () => { this.layernorm.build([...idShape, this.embeddingSize]); });
        this.trainableWeights = this.trainableWeights.concat(this.layernorm.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.layernorm.nonTrainableWeights);
        this.built = true;
    }
    call([inputIds, segmentIds = undefined], { positionIds = undefined, training = true }) {
        return tf.tidy(() => {
            if ((!segmentIds)) {
                segmentIds = tf.zerosLike(inputIds);
            }
            if ((!positionIds)) {
                positionIds = tf.range(0, inputIds.shape[1], 1, inputIds.dtype);
                positionIds = tf.expandDims(positionIds);
            }
            let positionEmbeddings = tf.gather(this.positionEmbedding.read(), positionIds);
            positionEmbeddings = tf.tile(positionEmbeddings, [inputIds.shape[0], 1, 1]);
            let tokenTypeEmbeddings = tf.gather(this.segmentEmbedding.read(), segmentIds);
            let tokenEmbeddings = tf.gather(this.wordEmbedding.read(), inputIds);
            let embeddings = tf.addN([tokenEmbeddings, tokenTypeEmbeddings, positionEmbeddings]);
            embeddings = this.layernorm.apply(embeddings);
            embeddings = this.dropout.apply(embeddings, { training: training });
            return embeddings;
        });
    }
    getConfig() {
        const config = { vocabSize: this.vocabSize, maxPositions: this.maxPositions, typeVocabSize: this.typeVocabSize, embeddingSize: this.embeddingSize, hiddenDropoutRate: this.hiddenDropoutRate, initializerRange: this.initializerRange, epsilon: this.epsilon, name: this.name, ...this.kwargs };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape([idShape, segmentIdsShape = undefined]) {
        let ret = [...idShape];
        ret.push(this.embeddingSize);
        return ret;
    }
}
tf.serialization.registerClass(AlbertEmbedding);