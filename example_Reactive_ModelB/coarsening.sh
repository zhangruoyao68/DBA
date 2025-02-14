for n in {1..12}; do
    echo $n

    sed -i "/scratch/c\cd /scratch/gpfs/ruoyaoz/ModelB_DBA$n" coarsening.cmd;

    cp coarsening_analysis.cpp coarsening.cmd /scratch/gpfs/ruoyaoz/ModelB_DBA$n;
    cd /scratch/gpfs/ruoyaoz/ModelB_DBA$n;
    g++ coarsening_analysis.cpp;

    rm -rf size_dist.txt avg_radius.txt
    sbatch coarsening.cmd;

    cd /home/ruoyaoz/3D_dendritic_growth_PFM_SGPU/ModelB_reaction/;
done