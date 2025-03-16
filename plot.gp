set terminal svg size 640,480 background rgb 'white'
set output './docs/figures/多項式の一致.gnuplot.svg'

# x 軸の表示範囲を広げすぎるとY値が急激に発散してしまうので、ここでは -5 ~ 5 程度
set xrange [-5:5]
set yrange [*:*]

plot \
    ( x**5 - 15*x**3 + 10*x**2 - 20*x ) title "f(x)", \
    ( 0.1*x**5 - x**4 + 0.5*x**3 + 20 ) title "g(x)"