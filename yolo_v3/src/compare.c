#include <stdio.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

void validate_compare(char *filename, char *weightfile)
{
    int i = 0;
    network *net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(net, weightfile);
    }
    srand((unsigned int)time(0));

    list *plist = get_paths("data/compare.val.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size/2;
    free_list(plist);

    clock_t time;
    int correct = 0;
    int total = 0;
    int splits = 10;
    int num = (i+1)*N/splits - i*N/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = 20;
    args.n = num;
    args.m = 0;
    args.d = &buffer;
    args.type = COMPARE_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*N/splits - i*N/splits;
        char **part = paths+(i*N/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);
        int j,k;
        for(j = 0; j < val.y.rows; ++j){
            for(k = 0; k < 20; ++k){
                if(val.y.vals[j][k*2] != val.y.vals[j][k*2+1]){
                    ++total;
                    if((val.y.vals[j][k*2] < val.y.vals[j][k*2+1]) == (pred.vals[j][k*2] < pred.vals[j][k*2+1])){
                        ++correct;
                    }
                }
            }
        }
        free_matrix(pred);
        printf("%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct/total, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

typedef struct {
    network net;
    char *filename;
    int class;
    int classes;
    float elo;
    float *elos;
} sortable_bbox;

int total_compares = 0;
int current_class = 0;

int elo_comparator(const void*a, const void *b)
{
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    if(box1.elos[current_class] == box2.elos[current_class]) return 0;
    if(box1.elos[current_class] >  box2.elos[current_class]) return -1;
    return 1;
}

int bbox_comparator(const void *a, const void *b)
{
    ++total_compares;
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    network net = box1.net;
    int class   = box1.class;

    image im1 = load_image_color(box1.filename, net.w, net.h);
    image im2 = load_image_color(box2.filename, net.w, net.h);
    float *X  = calloc(net.w*net.h*net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(&net, X);
    
    free_image(im1);
    free_image(im2);
    free(X);
    if (predictions[class*2] > predictions[class*2+1]){
        return 1;
    }
    return -1;
}

void bbox_update(sortable_bbox *a, sortable_bbox *b, int class, int result)
{
    int k = 32;
    float EA = 1.f/(1+powf(10, (b->elos[class] - a->elos[class])/400.f));
    float EB = 1.f/(1+powf(10, (a->elos[class] - b->elos[class])/400.f));
    float SA = result ? 1.f : 0;
    float SB = result ? 0 : 1.f;
    a->elos[class] += k*(SA - EA);
    b->elos[class] += k*(SB - EB);
}

void bbox_fight(network net, sortable_bbox *a, sortable_bbox *b, int classes, int class)
{
    image im1 = load_image_color(a->filename, net.w, net.h);
    image im2 = load_image_color(b->filename, net.w, net.h);
    float *X  = calloc(net.w*net.h*net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(&net, X);
    ++total_compares;

    int i;
    for(i = 0; i < classes; ++i){
        if(class < 0 || class == i){
            int result = predictions[i*2] > predictions[i*2+1];
            bbox_update(a, b, i, result);
        }
    }
    
    free_image(im1);
    free_image(im2);
    free(X);
}
