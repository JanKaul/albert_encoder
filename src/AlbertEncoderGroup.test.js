import { expect } from '@esm-bundle/chai';
import { tf, AlbertEncoderGroup } from "@weblab-notebook/albert";

describe("test encoder group", () => {
    it('', () => {
        let testEncoderGroup = new AlbertEncoderGroup({ numLayersEachGroup: 1 });
        let hiddenState = tf.randomUniform([2, 16, 768]);
        let mask = tf.tensor2d([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], [2, 16], "int32");
        let [output, allOutputs, allWeights] = testEncoderGroup.apply([hiddenState, mask]);
        expect(output.shape).to.eql([2, 16, 768]);
        for (const out of allOutputs) {
            expect(out.shape).to.eql([2, 16, 768]);
        }
        for (const weights of allWeights) {
            expect(weights.shape).to.eql([2, 16, 12, 16]);
        }
    })
});

describe('test masked attention', () => {
    it('', () => {
        let testLayer = new AlbertEncoderGroup({ numLayersEachGroup: 1, hiddenSize: 8, numAttentionHeads: 2 });
        let batch_size = 3;
        let query = tf.input({ shape: [4, 8], batchSize: batch_size });
        let mask_tensor = tf.input({ shape: [4], batchSize: batch_size });
        let [output1, one] = testLayer.apply([query, mask_tensor]);
        let model1 = tf.model({ inputs: [query, mask_tensor], outputs: output1 });
        let from_data = tf.scalar(10).mul(tf.rand([batch_size, 4, 8], Math.random));
        let mask_data = tf.rand([batch_size, 4], () => {
            let min = Math.ceil(0);
            let max = Math.floor(2);
            return Math.floor(Math.random() * (max - min) + min);
        });
        let null_mask_data = tf.ones([batch_size, 4]);
        let masked_output_data1 = model1.predictOnBatch([from_data, mask_data]);
        let unmasked_output_data1 = model1.predictOnBatch([from_data, null_mask_data]);
        let result1 = tf.mean(tf.abs(tf.sqrt(tf.squaredDifference(masked_output_data1, unmasked_output_data1)).div(unmasked_output_data1)));
        expect(result1.arraySync()).to.be.greaterThan(0.01);
    })
});