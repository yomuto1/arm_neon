#include "darknet.h"

static FILE *fp;

unsigned char ga_sized_u08[416 * 416 * 3];

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
	int i_s32, j_s32, k_s32;
	float tmp_f32;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45f;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);

		/* hyuk, integer version */
		for (k_s32 = 0; k_s32 < sized.c; k_s32++)
		{
			for (j_s32 = 0; j_s32 < sized.h; j_s32++)
			{
				for (i_s32 = 0; i_s32 < sized.w; i_s32++)
				{
					tmp_f32 = sized.data[i_s32 + j_s32 * sized.w + k_s32 * sized.w * sized.h] * 255.f;
					ga_sized_u08[i_s32 + j_s32 * sized.w + k_s32 * sized.w * sized.h] = tmp_f32 > 255.f ? 255.f : (unsigned char)tmp_f32;
				}
			}
		}
		{
			FILE *ffp;

			ffp = fopen("input_image_u08_416x416x3.yuv", "wb");

			fwrite(ga_sized_u08, sizeof(unsigned char), sized.w * sized.h * sized.c, ffp);

			fclose(ffp);
		}

        layer l = net->layers[net->n-1];

        float *X = sized.data;

		fp = fopen("performance_results.ini", "w");

        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: network_predict 1st %f sec\n", input, what_time_is_it_now()-time);

		int i;
		time = what_time_is_it_now();
		for (i = 0; i < 2; i++)
		{
			network_predict(net, X);
		}
		printf("%s: network_predict 2 times %f sec\n", input, what_time_is_it_now() - time);

		int nboxes = 0;
		time = what_time_is_it_now();
		//detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		printf("%s: post processing %f sec\n", input, what_time_is_it_now() - time);

		fclose(fp);

		//draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        //free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
