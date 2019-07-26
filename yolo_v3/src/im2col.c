#include "im2col.h"

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;
	int im_row, im_col;
	float tmp_dat_f32;

	for (c = 0; c < channels_col; ++c)
	{
        const int w_offset = c % ksize;
        const int h_offset = (c / ksize) % ksize;
        const int c_im = c / ksize / ksize;

        for (h = 0; h < height_col; ++h)
		{
			float * __restrict p_tmp_data_col_f32 = &data_col[(c * height_col + h) * width_col];
			const float * __restrict p_tmp_data_im_f32 = &data_im[width * ((h_offset + h * stride - pad) + height * c_im)];

			im_row = h_offset + h * stride - pad;

			for (w = 0; w < width_col; ++w)
			{
                im_col = w_offset + w * stride - pad;

				if ((im_row >= 0) && (im_col >= 0) && (im_row < height) && (im_col < width))
				{
					tmp_dat_f32 = p_tmp_data_im_f32[im_col];
				}
				else
				{
					tmp_dat_f32 = 0;
				}

				*p_tmp_data_col_f32++ = tmp_dat_f32;
            }
        }
    }
}

