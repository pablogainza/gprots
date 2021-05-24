cd uniref/
for dir in *
do
    cd $dir
    in_fasta=$(ls cluster*)
    python3 ../../source/embeddings/make_embeddings.py $in_fasta in_data/
    cd ..
done
