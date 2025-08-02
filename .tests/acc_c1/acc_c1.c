#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main( int argc, char* argv[] )
{
    int n;           /* size of the vector */
    float *a;        /* the vector */
    float *restrict r; /* the results */
    float *e;        /* expected results */
    int i;

    if( argc > 1 )
        n = atoi( argv[1] );
    else
        n = 100000;
    if( n <= 0 ) n = 100000;

    a = (float*)malloc(n * sizeof(float));
    r = (float*)malloc(n * sizeof(float));
    e = (float*)malloc(n * sizeof(float));

    /* initialize */
    // for( i = 0; i < n; ++i ) a[i] = (float)(i + 1);

    /*
     * a[0:n] 表示数组 a 从索引 0 开始的 n 个元素。
     * copyin(a[0:n]): 将 'a' 从 CPU 复制到 GPU。
     * copyout(r[0:n]): 将 'r' 从 GPU 复制回 CPU。
     */
    #pragma acc kernels loop copyin(a[0:n]) copyout(r[0:n])
    for( i = 0; i < n; ++i )
    {
        r[i] = a[i] * 2.0f;
    }

    /* compute on the host to compare */
    // for( i = 0; i < n; ++i )
    // {
    //     e[i] = a[i] * 2.0f;
    // }

    /* check the results */
    // for( i = 0; i < n; ++i )
    // {
    //     assert( r[i] == e[i] );
    // }

    // printf( "%d iterations completed\n", n );

    free(a);
    free(r);
    free(e);

    return 0;
}