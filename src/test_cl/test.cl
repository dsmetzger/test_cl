
#define MAXSIZE (4096)

#define M_PI (3.14159)


void convolve(global float* x, global float* h, global float* y, int const h_size) {
    *y=0;
    for (int i=0; i< h_size; ++i){
        *y += h[i]*x[i];
    }
}

kernel void sum(global const float* a, global const float* b, global float* result, int const size) {
    const int itemId = get_global_id(0);

    if(itemId < size-1024) {
        convolve( &a[itemId], &b[0], &result[itemId], 1024);
    }
    
    /*
    local float scratch[MAXSIZE];
    if(itemId < size) {
        scratch[itemId] = a[itemId] * 4.0;
        result[itemId] = b[itemId] + scratch[itemId];
        for (int x=0; x< 1024; ++x){
            result[itemId] = b[itemId] + result[itemId];
        }
    }
    */
}

