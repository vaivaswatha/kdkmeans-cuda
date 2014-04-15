string=""
for plot in /tmp/*.plot
do
    string="${string},\"$plot\""
done

string=`cut -c 2- <<EOF
$string
EOF`

echo "set key off" > /tmp/plot
echo "plot $string" >> /tmp/plot
gnuplot -persist < /tmp/plot

# ah
