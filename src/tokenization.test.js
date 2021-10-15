import { expect } from '@esm-bundle/chai';
import { tokenizer, cleanText } from "@weblab-notebook/albert";

let text = "This is some test text.";
let cleaned = cleanText(text);

let processor = await tokenizer();

let ids = await processor.encodeIds(cleaned);

it('encode ids', () => {
    expect(ids).to.eql(new Int32Array([48, 25, 109, 1289, 1854, 9]));
});

let pieces = await processor.decodeIds(ids);

it('decode ids', () => {
    expect(pieces).to.eql("this is some test text.");
});
