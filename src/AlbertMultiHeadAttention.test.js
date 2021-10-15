import { expect } from '@esm-bundle/chai';
import { tf, AlbertMultiHeadAttention } from "@weblab-notebook/albert_encoder";

describe("create attention layer without mask", () => {
    it('', () => {
        let testLayer = new AlbertMultiHeadAttention({ hiddenSize: 128, numAttentionHeads: 2 });
        let query = tf.input({ shape: [40, 80] });
        let value = tf.input({ shape: [20, 80] });
        let [output, weights] = testLayer.apply([query, value]);
        expect(output.shape).to.eql([null, 40, 80]);
    })
});

describe("create self attention layer without mask", () => {
    it('', () => {
        let testLayer = new AlbertMultiHeadAttention({ hiddenSize: 128, numAttentionHeads: 2 });
        let query = tf.input({ shape: [40, 80] });
        let [output, weights] = testLayer.apply([query, query]);
        expect(output.shape).to.eql([null, 40, 80]);
    })
});

describe('test masked attention', () => {
    it('', () => {
        let testLayer = new AlbertMultiHeadAttention({ hiddenSize: 8, numAttentionHeads: 2 });
        let batch_size = 3;
        let query = tf.input({ shape: [4, 8], batchSize: batch_size });
        let value = tf.input({ shape: [2, 8], batchSize: batch_size });
        let mask_tensor = tf.input({ shape: [4, 2], batchSize: batch_size });
        let [output1, one] = testLayer.apply([query, value, value, mask_tensor]);
        let model1 = tf.model({ inputs: [query, value, mask_tensor], outputs: output1 });
        let from_data = tf.scalar(10).mul(tf.rand([batch_size, 4, 8], Math.random));
        let to_data = tf.scalar(10).mul(tf.rand([batch_size, 2, 8], Math.random));
        let mask_data = tf.rand([batch_size, 4, 2], () => {
            let min = Math.ceil(0);
            let max = Math.floor(2);
            return Math.floor(Math.random() * (max - min) + min);
        });
        let null_mask_data = tf.ones([batch_size, 4, 2]);
        let masked_output_data1 = model1.predictOnBatch([from_data, to_data, mask_data]);
        let unmasked_output_data1 = model1.predictOnBatch([from_data, to_data, null_mask_data]);
        let result1 = tf.mean(tf.abs(tf.sqrt(tf.squaredDifference(masked_output_data1, unmasked_output_data1)).div(unmasked_output_data1)));
        expect(result1.arraySync()).to.be.greaterThan(0.01);
        let key = new tf.input({ shape: [2, 8] });
        let [output2, two] = testLayer.apply([query, value, key, mask_tensor]);
        let model2 = tf.model({ inputs: [query, value, key, mask_tensor], outputs: output2 });
        let masked_output_data2 = model2.predictOnBatch([from_data, to_data, to_data, mask_data]);
        let unmasked_output_data2 = model2.predictOnBatch([from_data, to_data, to_data, null_mask_data]);
        let result2 = tf.mean(tf.abs(tf.sqrt(tf.squaredDifference(masked_output_data2, unmasked_output_data2)).div(unmasked_output_data2)));
        expect(result2.arraySync()).to.be.greaterThan(0.01);
    })
});