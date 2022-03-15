eval "$(conda shell.bash hook)"
conda activate embedding

project_path="/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/genome-structure-function-aggregation"
input_dir="${project_path}/data/K562/Hi-C/hg38/oe_25000"
output_dir="${project_path}/data/K562/Hi-C/hg38/oe_25000_weibull"
log_dir="results/weibull_log"
p_value_code="${project_path}/utilities/p_value_calc.py"
for ((i=1;i<=22;i++)); do
	for ((j=$((i));j<=22;j++)); do
        input_file="${input_dir}/chr${i}_chr${j}.txt"
        python ${p_value_code} -i ${input_file} -l ${log_dir} -f ${i} -s ${j} -o ${output_dir}
    done
done
