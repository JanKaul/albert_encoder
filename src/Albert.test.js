import { expect } from '@esm-bundle/chai';
import { tf, Albert } from "@weblab-notebook/albert_encoder";

describe("test encoder", () => {
    it('', () => {
        let ids = tf.input({ shape: [8,], dtype: "int32" });
        let segmentIds = tf.input({ shape: [8,], dtype: "int32" });
        let attentionMask = tf.input({ shape: [8,], dtype: "int32" });
        let next = new Albert({ vocabSize: 100, maxPositions: 8 });
        let [first, output] = next.apply([ids, segmentIds, attentionMask]);
        let model = tf.model({ inputs: [ids, segmentIds, attentionMask], outputs: output });

        let batchSize = 2;
        let inputIds = tf.rand([batchSize, 8], () => {
            let min = Math.ceil(0);
            let max = Math.floor(100);
            return Math.floor(Math.random() * (max - min) + min);
        }, "int32");
        let inputSegmentIds = tf.zerosLike(inputIds);
        let inputAttentionMask = tf.zerosLike(inputIds);
        let inputNoMask = tf.ones([batchSize, 8]);
        let result1 = model.predictOnBatch([inputIds, inputSegmentIds, inputAttentionMask]);
        let result2 = model.predictOnBatch([inputIds, inputSegmentIds, inputNoMask]);
        let result = tf.mean(tf.abs(tf.sqrt(tf.squaredDifference(result1, result2)).divNoNan(result2)));
        expect(result.arraySync()).to.be.greaterThan(0.01);
    })
});



describe("test training model", () => {
    it('', (done) => {
        let ids = tf.input({ shape: [8,], dtype: "int32" });
        let segmentIds = tf.input({ shape: [8,], dtype: "int32" });
        let attentionMask = tf.input({ shape: [8,], dtype: "int32" });
        let next = new Albert({ vocabSize: 100, maxPositions: 8, hiddenSize: 128, numAttentionHeads: 2 });
        let [first, pooledOutput] = next.apply([ids, segmentIds, attentionMask]);
        let dense = tf.layers.dense({ units: 2 });
        let output = dense.apply(pooledOutput);
        let model = tf.model({ inputs: [ids, segmentIds, attentionMask], outputs: output });

        let batchSize = 2;
        let inputIds = tf.rand([batchSize, 8], () => {
            let min = Math.ceil(0);
            let max = Math.floor(100);
            return Math.floor(Math.random() * (max - min) + min);
        }, "int32");
        let inputSegmentIds = tf.zerosLike(inputIds);
        let inputAttentionMask = tf.zerosLike(inputIds);
        let inputNoMask = tf.ones([batchSize, 8]);
        let predictions = tf.rand([batchSize, 2], () => {
            let min = Math.ceil(0);
            let max = Math.floor(1);
            return Math.floor(Math.random() * (max - min) + min);
        }, "int32");
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        model.trainOnBatch([inputIds, inputSegmentIds, inputNoMask], predictions).then(_ => {
            let result1 = model.predictOnBatch([inputIds, inputSegmentIds, inputAttentionMask]);
            let result2 = model.predictOnBatch([inputIds, inputSegmentIds, inputNoMask]);
            let result = tf.mean(tf.abs(tf.sqrt(tf.squaredDifference(result1, result2)).divNoNan(result2)));
            expect(result.arraySync()).to.be.greaterThan(0.01);
            done()
        }).catch(err => done(err));
    })
});