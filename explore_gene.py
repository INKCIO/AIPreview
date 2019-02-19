import numpy
import pandas

with open('data/counts.txt', 'rt') as f:
    data_table = pandas.read_csv(f, index_col=0)
    print(data_table.iloc[:5, :5])

samples = list(data_table.columns)

with open('data/genes.csv', 'rt') as g:
    gene_info = pandas.read_csv(g, index_col=0)
    print(gene_info.iloc[:5])

print('Genes in data_table: ', data_table.shape)
print('Genes in gen_info: ', gene_info.shape)

matched_index = pandas.Index.intersection(data_table.index, gene_info.index)
counts = numpy.asarray(data_table.loc[matched_index], dtype=int)

gene_names = numpy.array(matched_index)
gene_length = numpy.asarray(
    gene_info.loc[matched_index]['GeneLength'], dtype=int)

