// make a curve of the performance of all solver files based on the length of the system
set title "Temps passé par rapport à la taille du système pour les 3 programmes"
set xlabel "Len of the system"
set ylabel "Time (s)"
set grid

plot "output_ite.dat" using 1:2 with lines title "Iteratif", \
     "output_omp.dat" using 1:2 with lines title "OpenMP", \
     "output_cuda.dat" using 1:2 with lines title "CUDA"