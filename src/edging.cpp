template<typename T>
void edging(T* u , T lambda , T nu , int M, int N, int B) f
for ( int k = 0 ; k < B; k++) {
for ( int i = 0 ; i < N; i++) {
for ( int j = 0 ; j < M; j++) {
int X = j + i * M + k * N * M;
T dx = i + 1 < N ? u[ j + ( i +1) * M + k * N * M] - u [X] : 0 ;
T dy = j + 1 < M ? u[ j + 1 + i * M + k * N * M] - u [X] : 0 ;
T norm = sqrt ( dx*dx+dy*dy ) ;
T factor = sqrt (nu/lambda ) ;
if (norm >= factor ) {
T c = (T) 1 / log( sqrt( 2 ) / factor ) ;
u[X] = (1 - c * log(norm / factor ) ) * u [X] ;
}
}
}
}
