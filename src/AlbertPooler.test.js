import { expect } from '@esm-bundle/chai';
import { tf, AlbertPooler } from "@weblab-notebook/albert_encoder";

describe("test encoder", () => {
    it('', () => {
        let testPooler = new AlbertPooler({ hiddenSize: 768 });
        let embedding = tf.randomUniform([2, 16, 768]);
        let output = testPooler.apply([embedding]);
        expect(output.shape).to.eql([2, 768]);
    })
});

describe("test encoder", () => {
    it('', () => {
        let testPooler = new AlbertPooler({ hiddenSize: 768 });
        let embedding = tf.input({ shape: [16, 768] });

        let output = testPooler.apply([embedding]);

        let model = tf.model({ inputs: embedding, outputs: output });

        let inputEmbedding = tf.randomUniform([2, 16, 768]);

        let result = model.predictOnBatch(inputEmbedding);
        expect(result.shape).to.eql([2, 768]);
    })
});