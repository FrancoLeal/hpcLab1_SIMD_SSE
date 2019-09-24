#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <xmmintrin.h>
#include <time.h>


//Funcion que lee la imagen y la guarda en dos matrices
//Entradas: nombre del archivo, tamaño de la imagen, matriz de enteros, matriz de flotantes
//Salidas: matriz de enteros y matriz de flotantes con la imagen
void readImage(char * filename,int n, int ** imageS, float ** imageP){
    FILE * fp = fopen(filename,"rb");
    if(fp==NULL){
        printf("No es posible abrir el archivo.\n");
        fclose(fp);
        exit(1);
    }
    int * buffer;
    buffer = (int*)malloc(sizeof(int)*n);
    int i,j;
    i=0;
    while(!feof(fp)){
        int read = fread(buffer,sizeof(int),n,fp);
        if(read!=0){
            for(j=0;j<n;j++){
                imageS[i][j] = buffer[j];
                imageP[i][j] = (float)buffer[j];
            }
            i++;
        }
    }
    fclose(fp);
    free(buffer);
}

//Funcion que aplica dilation secuencial
//Entradas: imagen original, imagen de salida, tamaño de la imagen
//Salidas: imagen con el operador morfologico aplicado
void secuentialDilation(int ** image,int ** out,int n){
    int i,j;
    for(j=0;j<n;j++){
        out[0][j]=image[0][j];
        out[n-1][j]=image[n-1][j];
    }
    for(i=0;i<n;i++){
        out[i][0]=image[i][0];
        out[i][n-1]=image[i][n-1];
    }
    for(i=1;i<n-1;i++){
        for(j=1;j<n-1;j++){
            if(image[i-1][j] == 255 || image[i][j-1] == 255  || image[i][j] == 255  || image[i][j+1] == 255 || image[i+1][j] == 255){
                out[i][j] = 255;
            }
            else{
                out[i][j] = 0;
            }
        }
    }
}
//Funcion que escribe el archivo de salida de la imagen secuencial
//Entradas: nombre del archivo de salida, imagen secuencial, tamaño de la imagen
//Salida: archivo de la imagen
void writeImageS(char * filename, int ** image, int n){
    FILE * fp = fopen(filename,"w");
    int * buffer;
    buffer = (int*)malloc(sizeof(int)*n);
    int i,j;
    for(i=0;i<n;i++){
        fwrite(image[i],sizeof(int),n,fp);
    }
    free(buffer);
    fclose(fp);
}
//Funcion que escribe el archivo de salida de la imagen con SIMD
//Entradas: nombre del archivo de salida, imagen con SIMD, tamaño de la imagen
//Salida: archivo de la imagen
void writeImageP(char * filename, float ** image, int n){
    FILE * fp = fopen(filename,"w");
    float * buffer;
    buffer = (float*)malloc(sizeof(float)*n);
    int i,j;
    for(i=0;i<n;i++){
        fwrite(image[i],sizeof(float),n,fp);
    }
    free(buffer);
    fclose(fp);
}
//Funcion que imprime por pantalla dos imagenes
//Entradas: imagen secuencial, imagen con SIMD, tamaño de la imagen
//SAlida: imagenes por consola en 0's y 1's
void printImages(int ** imageS, float ** imageP, int n){
    int i,j;
    printf("***************************\n");
    printf("*****Imagen secuencial*****\n");
    printf("***************************\n");
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            if(imageS[i][j]>0){
                printf(" 1 ");
            }
            else{
                printf(" 0 ");
            }
        }
        printf("\n");
    }
    printf("***************************\n");
    printf("******Imagen paralela******\n");
    printf("***************************\n");
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            if(imageP[i][j]>(float)0){
                printf(" 1 ");
            }
            else{
                printf(" 0 ");
            }
        }
        printf("\n");
    }

}

//Funcion que obtiene 4 elementos de una imagen a partir de una posición dada
//Entradas: imagen a buscar, posicion i inicial, posicion j inicial
//Salida: 4 elementos contiguos de la imagen

float * getElements(float ** image, int i, int j){
    int k,l=0;
    float * aux=(float*)malloc(sizeof(float)*4);
    for(k=j;k<j+4;k++){
        aux[l] = image[i][k];
        l++;
    }
    return aux;
}

//Funcion que aplica dilation con SIMD
//Entradas: imagen original, imagen de salida, tamaño de la imagen
//Salidas: imagen con el operador morfologico aplicado
void simdDilation(float ** image,float ** out, int n){
    __m128 R0,R1,R2,R3,R4;
    int i,j,k;
    for(j=0;j<n;j++){
        out[0][j]=image[0][j];
        out[n-1][j]=image[n-1][j];
    }
    for(i=0;i<n;i++){
        out[i][0]=image[i][0];
        out[i][n-1]=image[i][n-1];
    }
    for(i=1;i<n-1;i++){
        for(j=1;j<n-4;j+=4){
            float * r0Array  __attribute__ ((aligned (16))) = getElements(image,i-1,j);
            R0 = _mm_load_ps(r0Array);
            float * r1Array  __attribute__ ((aligned (16))) = getElements(image,i,j-1);
            R1 = _mm_load_ps(r1Array);
            float * r2Array  __attribute__ ((aligned (16))) = getElements(image,i,j+1);
            R2 = _mm_load_ps(r2Array);
            float * r3Array  __attribute__ ((aligned (16))) = getElements(image,i+1,j);
            R3 = _mm_load_ps(r3Array);
            float * r4Array  __attribute__ ((aligned (16))) = getElements(image,i,j);
            R4 = _mm_load_ps(r4Array);
            __m128 maximum = _mm_max_ps(R0,R1);
            maximum = _mm_max_ps(maximum,R2);
            maximum = _mm_max_ps(maximum,R3);
            maximum = _mm_max_ps(maximum,R4);
            for(k=0;k<4;k++){
                if(maximum[k]==255){
                    out[i][j+k]= (float)255;
                }
                else{
                    out[i][j+k]= (float)0;
                }
            }
            free(r0Array);
            free(r1Array);
            free(r2Array);
            free(r3Array);
            free(r4Array);
        }
    }
}

int main(int argc, char **argv){
    if(argc<4){
        printf("Argumentos insuficientes.\n");
        exit(1);
    }
    /*
    i: nombre imagen de entrada
    s: nombre imagen salida secuencial
    p: nombre imagen salida SIMD
    N: ancho de la imagen de entrada
    D: Imprimir imagen por consola
    */
    
    char * inputImage;
    char * outputImageS;
    char * outputImageP;
    int n;
    int d=0;

    int c;


    opterr=0;

    while((c=getopt(argc,argv,"i:s:p:N:D"))!=-1){
        switch (c){
            case 'i':
                inputImage = optarg;
                break;
            case 's':
                outputImageS = optarg;
                break;
            case 'p':
                outputImageP = optarg;
                break;
            case 'N':
                sscanf(optarg,"%d",&n);
                break;
            case 'D':
                d = 1;
                break;
            case '?':
                if (optopt == 'i' || optopt == 's' || optopt == 'p' || optopt == 'N') {
                    fprintf (stderr, "Opcion -%c requiere un argumento.\n", optopt);
                }
                else if (isprint (optopt)) {
                    fprintf (stderr, "Opcion desconocida -%c.\n", optopt);
                }
                else {
                    fprintf (stderr, "Caracter desconocido `\\x%x'.\n", optopt);

                }
                return 1;
		    default:
			    abort ();
        }
    }

    printf("input: %s, output1: %s, output2: %s, n: %d, d: %d\n",inputImage,outputImageS,outputImageP,n,d);
    int i;
    int ** originalImageS;
    originalImageS = (int**)malloc(sizeof(int*)*n);
    if(originalImageS==NULL){
        printf("Error de memoria\n");
        exit(1);
    }
    for(i=0;i<n;i++){
        originalImageS[i]=(int*)malloc(sizeof(int)*n);
        if(originalImageS[i]==NULL){
            printf("Error de memoria\n");
            exit(1);
        }
    }
    float ** originalImageP;
    originalImageP = (float**)malloc(sizeof(float*)*n);
    if(originalImageP==NULL){
        printf("Error de memoria\n");
        exit(1);
    }
    for(i=0;i<n;i++){
        originalImageP[i]=(float*)malloc(sizeof(float)*n);
        if(originalImageP[i]==NULL){
            printf("Error de memoria\n");
            exit(1);
        }
    }
    readImage(inputImage,n,originalImageS,originalImageP);
    int ** secuentialImage;
    secuentialImage = (int**)malloc(sizeof(int*)*n);
    if(secuentialImage==NULL){
        printf("Error de memoria\n");
        exit(1);
    }
    for(i=0;i<n;i++){
        secuentialImage[i]=(int*)malloc(sizeof(int)*n);
        if(secuentialImage[i]==NULL){
            printf("Error de memoria\n");
            exit(1);
        }
    }
    clock_t startS,startP;
    startS=clock();
    secuentialDilation(originalImageS,secuentialImage,n);
    printf("Secuencial: %.16g segundos.\n",(double)(clock()-startS)/CLOCKS_PER_SEC);
    writeImageS(outputImageS,secuentialImage,n);
    float ** simdImage;
    simdImage = (float**)malloc(sizeof(float*)*n);
    if(simdImage==NULL){
        printf("Error de memoria\n");
        exit(1);
    }
    for(i=0;i<n;i++){
        simdImage[i]=(float*)malloc(sizeof(float)*n);
        if(simdImage[i]==NULL){
            printf("Error de memoria\n");
            exit(1);
        }
    }
    startP=clock();
    simdDilation(originalImageP,simdImage,n);
    printf("SIMD: %.16g segundos.\n",(double)(clock()-startP)/CLOCKS_PER_SEC);
    writeImageP(outputImageP,simdImage,n);
    if(d==1){
        printImages(secuentialImage,simdImage,n);
    }
    for(i=0;i<n;i++){
        free(originalImageS[i]);
    }
    free(originalImageS);
    for(i=0;i<n;i++){
        free(originalImageP[i]);
    }
    free(originalImageP);
    for(i=0;i<n;i++){
        free(secuentialImage[i]);
    }
    free(secuentialImage);
    for(i=0;i<n;i++){
        free(simdImage[i]);
    }
    free(simdImage);
    
}