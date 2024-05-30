// make a curve of the performance of all solver files based on the nb on the length of the system
set title "Time Data from Three Programs"
set xlabel "Len of the system"
set ylabel "Time (s)"
set grid

plot "output1.dat" using 1:2 with lines title "Program 1", \
     "output2.dat" using 1:2 with lines title "Program 2", \
     "output3.dat" using 1:2 with lines title "Program 3"