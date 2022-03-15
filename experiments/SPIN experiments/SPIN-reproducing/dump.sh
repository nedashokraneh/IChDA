juicer_path="/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/genome-structure-function-aggregation/utilities/juicer_tools_1.22.01.jar"
hic_path="/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/genome-structure-function-aggregation/data/K562/hg38/Hi-C/4DNFITUOMFUQ.hic"
dump_dir="/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/genome-structure-function-aggregation/data/K562/hg38/Hi-C"
resolution=25000
type="oe"
norm="KR"
dump_dir="${dump_dir}/${type}_${resolution}_${norm}"
mkdir ${dump_dir}
for ((i=1;i<=22;i++)); do
	for ((j=$((i));j<=22;j++)); do
		c1=chr${i}
		c2=chr${j}
        	echo "extracting Hi-C of ${c1} and ${c2}"
        	dump_path="${dump_dir}/${c1}_${c2}.txt"
		java -jar ${juicer_path} dump oe ${norm} ${hic_path} ${i} ${j}  BP ${resolution} ${dump_path}
	done
done
