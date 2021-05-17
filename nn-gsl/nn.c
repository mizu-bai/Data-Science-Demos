/**
 * nn.c
 * 
 * Created by mizu-bai
 * 
 * This is a implementation of Tensors Warm-up: numpy in LEARNING PYTORCH WITH EXAMPLES
 * (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#) using C with GSL.
 */


#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

int main(int argc, const char *argv[]) {
    // create random input and output data
    int size = 2000;
    gsl_vector *x = gsl_vector_alloc(size);
    gsl_vector *y = gsl_vector_alloc(size);
    for (int i = 0; i < size; i++) {
        double x_tmp = 2 * M_PI / (double)(size - 1) * (double)i - M_PI;
        gsl_vector_set(x, i, x_tmp);
        gsl_vector_set(y, i, sin(x_tmp));
    }

    // randomly init weights
    int w_size = 4;
    gsl_vector *w = gsl_vector_alloc(w_size);
    gsl_rng *r = gsl_rng_alloc(gsl_rng_default);
    for (int i = 0; i < w_size; i++) {
        gsl_vector_set(w, i, gsl_ran_gaussian(r, 1.0));
    }
    gsl_rng_free(r);

    double learning_rate = 1e-6;
    for(int t = 0; t < 2000; t++) {
        // forward pass, compute predicted y & compute loss
        gsl_vector *y_pred = gsl_vector_alloc(size);
        double loss = 0;
        for (int i = 0; i < size; i++) {
            double w_tmp = gsl_poly_eval(w->data, w_size, gsl_vector_get(x, i));
            gsl_vector_set(y_pred, i, w_tmp);
            loss += gsl_pow_2(gsl_vector_get(y_pred, i) - gsl_vector_get(y, i));
        }
        if (t % 100 == 99) {
            printf("t: %d, loss: %.4f\n", t, loss);
        }

        // compute gradients of w with respect to loss
        gsl_vector *grad_y_pred = gsl_vector_alloc(size);
        gsl_vector_memcpy(grad_y_pred, y_pred);
        gsl_vector_sub(grad_y_pred, y);
        gsl_vector_scale(grad_y_pred, 2);
        gsl_vector *grad_w = gsl_vector_alloc(w_size);
        gsl_vector_set_all(grad_w, 0.0);
        for(int i = 0; i < size; i++) {
            gsl_vector *grad_y_pred_tmp = gsl_vector_alloc(size);
            gsl_vector *x_tmp = gsl_vector_alloc(size);
            
            gsl_vector_set(grad_w, 0, gsl_vector_get(grad_w, 0) + gsl_vector_get(grad_y_pred, i));

            gsl_vector_memcpy(grad_y_pred_tmp, grad_y_pred);
            gsl_vector_mul(grad_y_pred_tmp, x);
            gsl_vector_set(grad_w, 1, gsl_vector_get(grad_w, 1) + gsl_vector_get(grad_y_pred_tmp, i));
            
            gsl_vector_memcpy(grad_y_pred_tmp, grad_y_pred);
            gsl_vector_memcpy(x_tmp, x);
            gsl_vector_mul(x_tmp, x);
            gsl_vector_mul(grad_y_pred_tmp, x_tmp);
            gsl_vector_set(grad_w, 2, gsl_vector_get(grad_w, 2) + gsl_vector_get(grad_y_pred_tmp, i));

            gsl_vector_memcpy(grad_y_pred_tmp, grad_y_pred);
            gsl_vector_memcpy(x_tmp, x);
            gsl_vector_mul(x_tmp, x);
            gsl_vector_mul(x_tmp, x);
            gsl_vector_mul(grad_y_pred_tmp, x_tmp);
            gsl_vector_set(grad_w, 3, gsl_vector_get(grad_w, 3) + gsl_vector_get(grad_y_pred_tmp, i));
            
            gsl_vector_free(grad_y_pred_tmp);
            gsl_vector_free(x_tmp);
        }

        gsl_vector_scale(grad_w, learning_rate);
        gsl_vector_sub(w, grad_w);

        // free
        gsl_vector_free(y_pred);
        gsl_vector_free(grad_y_pred);
        gsl_vector_free(grad_w);
    }

    printf("Result: y = %.4f + %.4f * x + %.4f * x^2 + %.4f * x^3\n", gsl_vector_get(w, 0), gsl_vector_get(w, 1), gsl_vector_get(w, 2), gsl_vector_get(w, 3));

    // free
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(w);

    return 0;
}