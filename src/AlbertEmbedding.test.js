import { expect } from '@esm-bundle/chai';
import { tf, AlbertEmbedding } from "@weblab-notebook/albert";

describe("test embedding", () => {
    it('', () => {
        let testEmbed = new AlbertEmbedding({ vocabSize: 100, embeddingSize: 128 });
        let inputIds = tf.tensor2d([[0, 2, 3, 4, 5, 1], [0, 2, 3, 4, 5, 1]], [2, 6], "int32");
        let typeIds = tf.tensor2d([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], [2, 6], "int32");
        let emb = testEmbed.apply([inputIds, typeIds]);
        expect(emb.shape).to.eql([2, 6, 128]);
    })
});

describe("test embedding", () => {
    it('', () => {
        let testEmbed = new AlbertEmbedding({ vocabSize: 100, embeddingSize: 128 });
        let batchSize = 2;
        let ids = tf.input({ shape: [6], batchSize: batchSize, dtype: "int32" });
        let types = tf.input({ shape: [6], batchSize: batchSize, dtype: "int32" });
        let emb = testEmbed.apply([ids, types]);
        let model = tf.model({ inputs: [ids, types], outputs: emb });

        let inputIds = tf.tensor2d([[0, 2, 3, 4, 5, 1], [0, 2, 3, 4, 5, 1]], [2, 6], "int32");
        let typeIds = tf.tensor2d([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], [2, 6], "int32");

        let result = model.predictOnBatch([inputIds, typeIds]);
        expect(result.shape).to.eql([2, 6, 128]);
    })
});