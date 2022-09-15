def generate_fithic_significant_interactions_file(self, base_count):

    out_file_path = os.path.join(self.output_interactions_dir_path, 'fithic_significant_interactions_b{}.txt'.format(base_count))
    chroms_valid_lengths = [len(self.get_valid_bins(c)) for c in np.arange(1,23)]
    min_length = np.min(chroms_valid_lengths)
    for c1 in np.arange(1,2):
        for c2 in np.arange(c1,2):
            c1_valid_length = chroms_valid_lengths[c1-1]
            c2_valid_length = chroms_valid_lengths[c2-1]
            ratio = (c1_valid_length/min_length) * (c2_valid_length/min_length)
            significant_contacts = int(base_count*ratio)
            print('adding {} contacts for chr{} and chr{}'.format(significant_contacts,c1,c2))
            fithic_df = self.get_fithic_significant_df(c1, c2, 0.1, significant_contacts)
            fithic_df['ind1'] = [self.pos2ind_dict['chr{}'.format(c1)][p] for p in fithic_df['pos1']]
            fithic_df['ind2'] = [self.pos2ind_dict['chr{}'.format(c2)][p] for p in fithic_df['pos2']]
            fithic_df[['ind1','ind2']] = fithic_df[['ind1','ind2']].astype(int)
            fithic_df['weight'] = 1
            fithic_df = fithic_df[fithic_df['ind1'] != fithic_df['ind2']]
            fithic_df.loc[:,['ind1','ind2','weight']].to_csv(out_file_path, sep = "\t", header = None, index = False, mode = 'a')


def generate_oe_significant_interactions_file(self, base_count):

    out_file_path = os.path.join(self.output_interactions_dir_path, 'oe_significant_interactions_b{}.txt'.format(base_count))
    chroms_valid_lengths = [len(self.get_valid_bins(c)) for c in np.arange(1,23)]
    min_length = np.min(chroms_valid_lengths)
    for c1 in np.arange(1,23):
        for c2 in np.arange(c1,23):
            c1_valid_length = chroms_valid_lengths[c1-1]
            c2_valid_length = chroms_valid_lengths[c2-1]
            ratio = (c1_valid_length/min_length) * (c2_valid_length/min_length)
            significant_contacts = int(base_count*ratio)
            print('adding {} contacts for chr{} and chr{}'.format(significant_contacts,c1,c2))
            hic_df = self.get_indexed_hic_df(c1,c2,'oe')
            hic_df = hic_df.nlargest(significant_contacts, 'weight')
            hic_df['weight'] = 1
            hic_df = hic_df[hic_df['ind1'] != hic_df['ind2']]
            hic_df.to_csv(out_file_path, sep = "\t", header = None, index = False, mode = 'a')
