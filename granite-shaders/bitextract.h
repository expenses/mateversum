/* Copyright (c) 2020-2022 Hans-Kristian Arntzen
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef BITEXTRACT_H_
#define BITEXTRACT_H_

uint uint_bitfieldExtract(uint data, int offset, int bits) {
	uint mask = (1 << bits) - 1;
	return (data >> offset) & mask;
}

// Taken from:
// https://github.com/KhronosGroup/BC6H-Decoder-WASM/blob/322615ff672508d08f4781588fad6938062df2e7/assembly/decoder.ts#L1128-L1136
int signExtend(uint width, int value) {
	uint shift = 32 - width;
	return (value << shift) >> shift;
}

int int_bitfieldExtract(int data, int offset, int bits) {
	int mask = (1 << bits) - 1;
	int masked = (data >> offset) & mask;

	return signExtend(bits, masked);
}

uvec3 uvec3_bitfieldExtract(uvec3 data, int offset, int bits) {
	return uvec3(
		uint_bitfieldExtract(data.x, offset, bits),
		uint_bitfieldExtract(data.y, offset, bits),
		uint_bitfieldExtract(data.z, offset, bits)
	);
}

ivec3 ivec3_bitfieldExtract(ivec3 data, int offset, int bits) {
	return ivec3(
		int_bitfieldExtract(data.x, offset, bits),
		int_bitfieldExtract(data.y, offset, bits),
		int_bitfieldExtract(data.z, offset, bits)
	);
}

ivec3 imix(ivec3 x, ivec3 y, bvec3 b) {
	return ivec3(
		b.x ? y.x : x.x,
		b.y ? y.y : x.y,
		b.z ? y.z : x.z
	);
}

// https://stackoverflow.com/a/9144870
uint reverse(uint x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
}

int extract_bits(uvec4 payload, int offset, int bits)
{
	int last_offset = offset + bits - 1;
	int result;

	if (bits <= 0)
		result = 0;
	else if ((last_offset >> 5) == (offset >> 5))
		result = int(uint_bitfieldExtract(payload[offset >> 5], offset & 31, bits));
	else
	{
		int first_bits = 32 - (offset & 31);
		int result_first = int(uint_bitfieldExtract(payload[offset >> 5], offset & 31, first_bits));
		int result_second = int(uint_bitfieldExtract(payload[(offset >> 5) + 1], 0, bits - first_bits));
		result = result_first | (result_second << first_bits);
	}
	return result;
}

int extract_bits_sign(uvec4 payload, int offset, int bits)
{
	int last_offset = offset + bits - 1;
	int result;

	if (bits <= 0)
		result = 0;
	else if ((last_offset >> 5) == (offset >> 5))
		result = int_bitfieldExtract(int(payload[offset >> 5]), offset & 31, bits);
	else
	{
		int first_bits = 32 - (offset & 31);
		int result_first = int(uint_bitfieldExtract(payload[offset >> 5], offset & 31, first_bits));
		int result_second = int_bitfieldExtract(int(payload[(offset >> 5) + 1]), 0, bits - first_bits);
		result = result_first | (result_second << first_bits);
	}
	return result;
}

int extract_bits_reverse(uvec4 payload, int offset, int bits)
{
	int last_offset = offset + bits - 1;
	int result;

	if (bits <= 0)
		result = 0;
	else if ((last_offset >> 5) == (offset >> 5))
		result = int(reverse(uint_bitfieldExtract(payload[offset >> 5], offset & 31, bits)) >> (32 - bits));
	else
	{
		int first_bits = 32 - (offset & 31);
		uint result_first = uint_bitfieldExtract(payload[offset >> 5], offset & 31, first_bits);
		uint result_second = uint_bitfieldExtract(payload[(offset >> 5) + 1], 0, bits - first_bits);
		result = int(reverse(result_first | (result_second << first_bits)) >> (32 - bits));
	}
	return result;
}

#endif