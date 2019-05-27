import itertools
import pandas as pd
import numpy as np

#############################################
# Input & Output
#############################################

def read_cost_matrix(file, pair='RU/BG'):
    # Read data from excel
    table = pd.read_excel(io=file)
    
    # Get row labels
    row_labels = table[pair]
    
    # Get column labels
    column_labels = table.columns
    
    # Drop uneccessary first column
    table.drop([pair], axis=1, inplace=True)
    
    # Create new dataframe using characters as index
    df = pd.DataFrame(data=table.values, index=row_labels.values, columns=column_labels.values[1:])
    
    return df


def read_data(file, sheets, drop_duplicates=False, remove_whitespace=False, header=0, index_col=None):
    # Read data from excel
    df = pd.read_excel(io=file, sheet_name=sheets, header=header, index_col=index_col)
    
    # Drop duplicates
    if drop_duplicates:
        df.drop_duplicates(keep='first', inplace=True)
    
    # Remove white space characters (on the left and right of the string)
    if remove_whitespace:
        for c in df.columns:
            df[c] = df[c].str.strip()
        
    return df


def store_results(path, foreign, native, data, data2, char_entropy, char_entropy2, surprisals, surprisals2, mod_surprisals, mod_surprisals2, probs, probs2, costs):
    
    # Create a dictionary of files
    files = {
        f'{foreign}-{native}': data, 
        f'{foreign}-char-entropy': char_entropy, 
        f'{native}-char-entropy': char_entropy2, 
        f'{foreign}-{native}-surprisals': surprisals,
        f'{foreign}-{native}-mod-surprisals': mod_surprisals,
        f'{foreign}-{native}-probabilities': probs,
        f'{native}-{foreign}': data2,
        f'{native}-{foreign}-surprisals': surprisals2,
        f'{native}-{foreign}-mod-surprisals': mod_surprisals2,
        f'{native}-{foreign}-probabilities': probs2,
        'costs': costs
        }

    # Write files to disk
    _write_to_excel(files, path)


def _write_to_excel(dfs, file):        
    
    excel_writer = pd.ExcelWriter(file)
    
    for key, value in dfs.items():
        value.to_excel(excel_writer, key, na_rep='-')
    
    excel_writer.save()


#############################################
# Levenshtein distance and alignments
#############################################

def levenshtein_distance(df, foreign, native, costs):
    """
    Copute Levenshtein distance and alignments 
    in speaker - listener direction.

    Parameters
    ----------
    df : pandas dataframe
        dataframe used for computing Levenshtein distance.
    speaker : string
        speaker language.
    listener : string
        listener language.
    cost_matrix : pandas dataframe
        cost matrix used for computing alignment.

    Returns
    -------
    pandas dataframe
        a new dataframe containing Levenshtein distance and alignments for each word pair.

    Raises
    ------
    KeyError
        when the wrong cost matrix is passed. Speaker language should correspond to the rows of the cost matrix.
    """
    df = df.copy()
    return df.join((df.loc[:, [foreign, native]]).apply(func=lambda x: _needleman_wunsch(*x, subs_costs=costs), axis=1))


def _needleman_wunsch(source, target, delete_costs=1.0, insert_costs=1.0, subs_costs=None, verbose=False, get_lookup=False, swap=False):
    # Remove space
    source = source.replace(' ', '')
    target = target.replace(' ', '')
        
    def get_subs_cost(a, b):
        subs_cost = 1
        if subs_costs is not None:
            try:
                subs_cost = subs_costs.loc[a, b]
            except KeyError:
                print('Key error: ', (a, b))
        
        if verbose: print('Getting substitution costs for characters ({0}, {1}): {2}'.format(a, b, subs_cost))
        
        return subs_cost
    
    if verbose:
        print('Speaker word (source): {0}'.format(list(source)))
        print('Listener word (target): {0}'.format(list(target)))
   
    if verbose:
        print('Computing cost table')
    
    # Initialize cost table
    rows = len(target) + 1 
    cols = len(source) + 1 
    
    # Initialize first row and column
    M = np.zeros(shape=(rows, cols))
    
    
    for i in range(rows):  # initialize rows
        M[i, 0] = i
        
    for j in range(cols):  # initialize cols
        M[0, j] = j
    
    M[0,0] = 0
    
    if verbose:
        print(M)
        
    # Fill rest of the matrix recursively 
    for i in range(1, rows):
        for j in range(1, cols):
            _sub = M[i-1, j-1] + get_subs_cost(source[j-1], target[i-1])
            _ins = M[i-1, j] + 1 # up
            _del = M[i, j-1] + 1 # left
            M[i, j] = min(_sub, _ins, _del)
            
    if verbose:
        print('Distance matrix:')
        print(M)
        
    # Get edit distance
    ld = M[-1, -1]
    
    source_alignment = ''
    target_alignment = ''
    path = []
    
    i, j = rows - 1, cols - 1
    
    if verbose:
        print('Performing backtracking')
    
    # Perform backtracking to get alignment
    while i > 0 and j > 0:
                
        if M[i-1, j-1] == M[i-1, j] == M[i, j-1]:  # diagonal
            if verbose: print('All equal. Walk along diagonal')
            
            source_alignment += source[j-1]
            target_alignment += target[i-1]
            path.append(M[i, j])
            i, j = i-1, j-1
            
        elif M[i, j] == M[i-1, j-1] + get_subs_cost(source[j-1], target[i-1]):  # diagonal
            if verbose: print('Walk along diagonal')
            
            source_alignment += source[j-1]
            target_alignment += target[i-1]
            path.append(M[i, j])
            i, j = i-1, j-1
            
        else:
            if swap:
            
                if M[i, j] == M[i, j-1] + 1: # left
                    if verbose: print('Walk left')

                    source_alignment += source[j-1]
                    target_alignment += '-'
                    path.append(M[i, j])
                    j = j-1
        
                else: # up
                    if verbose: print('Walk up')

                    source_alignment += '-'
                    target_alignment += target[i-1]
                    path.append(M[i, j])
                    i = i-1
        
            else:
                if M[i, j] == M[i-1, j] + 1: # up
                    if verbose: print('Walk up')

                    source_alignment += '-'
                    target_alignment += target[i-1]
                    path.append(M[i, j])
                    i = i-1
                else: # left
                    if verbose: print('Walk left')

                    source_alignment += source[j-1]
                    target_alignment += '-'
                    path.append(M[i, j])
                    j = j-1
          
    while i > 0:
        source_alignment += '-'
        target_alignment += target[i-1]
        path.append(M[i, j])
        i -= 1
        
    while j > 0:
        source_alignment += source[j-1]
        target_alignment += '-'
        path.append(M[i, j])
        j -= 1
    
    # Get source and target alignments 
    source_alignment = source_alignment[::-1] 
    target_alignment = target_alignment[::-1] 

    # obtain normalized levenshtein distance by dividing by alignmed length
    normalized_ld = ld / len(source_alignment)  
    
    series = pd.Series(data=[len(list(source)), len(list(target)), source_alignment, target_alignment, len(list(source_alignment)), ld, normalized_ld], 
                       index=[f'foreign word length', f'native word length', f'foreign alignment', f'native alignment', 'alignment length', 'LD', 'normalized LD'])
    
    return series


#############################################
# Surprisal
#############################################


def character_surprisals(df, foreign, native, count_same=True, log=np.log2):
    
    # Create initial surprisal data frames
    foreign_characters = set()
    native_characters = set()
    
    # Find all characters occuring in source words
    for word in df[foreign]:
        foreign_characters.update(list(word))
    foreign_characters.add('-')
    foreign_characters = sorted(foreign_characters)
    
    # Find all characters occuring in target words
    for word in df[native]:
        native_characters.update(list(word))
    native_characters.add('-')
    native_characters = sorted(native_characters)
    
    # Create dataframe
    probs = pd.DataFrame(0, index=foreign_characters, columns=native_characters)
    surprisals = pd.DataFrame(np.nan, index=foreign_characters, columns=native_characters)

    # Compute surprisals based on alignments
    foreign_characters_dict = {key: [dict(), 0] for key in foreign_characters}
    
    # Iterate over all alignment pairs
    for i, row in df.iterrows():
        foreign_align = row['foreign alignment']
        native_align = row['native alignment']
        
        # Iterate over characters in the source alignment
        for j, c in enumerate(foreign_align):
            if c == native_align[j] and not count_same:
                pass
            else:
                # Collect alligned characters
                target_char = native_align[j]
                
                char_dict = foreign_characters_dict[c][0]
                if target_char in char_dict:
                    char_dict[target_char] += 1
                else:
                    char_dict[target_char] = 1
                           
    # Iterate over the mapping from source character to aligned target characters
    for key, values in foreign_characters_dict.items():
        for _, counts in values[0].items():
            values[1] += counts
        
        for char, counts in values[0].items():
            # Compute probability and surprisal
            probs.loc[key, char] =  counts / values[1]
            surprisals.loc[key, char] = log(1 / probs.loc[key, char])
        
    return probs, surprisals
    

def modify_character_surprisals(surprisals, diag_value=0.0):
    df = surprisals.copy()
    for i, r in enumerate(df.index):
        for j, c in enumerate(df.columns):
            if r == c:
                df.iloc[i, j] = diag_value
    return df


def word_adaptation_surprisal(df, char_surprisals, char_probs):
    df = df.copy()
    return df.join((df.loc[:, ['foreign alignment', 'native alignment']]).apply(func=lambda x: _compute_word_adaptation_surprisal(*x, surprisals=char_surprisals, probabilities=char_probs), axis=1))


def _compute_word_adaptation_surprisal(foreign_alignment, native_alignment, surprisals, probabilities):        
    keys = list(zip(foreign_alignment, native_alignment))
        
    surprisal = 0.0
    for r, c in keys:
        surprisal += surprisals.loc[r, c]
        
    df = pd.Series(data=[surprisal, surprisal / len(foreign_alignment)], index=['WAS', 'normalized WAS'])
        
    return df

#############################################
# Entropy
#############################################

def character_entropy(surprisals, probs):
    characters = surprisals.index
    entropies = []
    
    for c in characters:
        entropy = np.sum(surprisals.loc[c] * probs.loc[c])
        # print('character: {:s} entropy: {:.4f}'.format(c, entropy))
        entropies.append(entropy)
    
    df = pd.DataFrame(data=entropies, index=characters, columns=['entropy (per character)'])
    return df

def full_conditional_entropy(Y, X, aligned_words, surprisalsXY, probsXY):
    """Compute H(Y|X)"""
    
    text = ''.join(aligned_words['native alignment'])
    
    # chars = set(text)
    chars = surprisalsXY.index

    # Compute probability of seeing each char in language X
    X_char_prob = {}
    for c in chars:
        X_char_prob[c] = text.count(c) / len(text)

    # Compute conditional entropy H(Y|X=x)
    sumx = 0.0
    for c in chars: # sum over X
        sumy = np.sum(probsXY.loc[c] * surprisalsXY.loc[c]) # sum over Y
        sumx += X_char_prob[c] * sumy

    return sumx


#############################################
# Intelligibility
#############################################   

def append_intelligibility_scores(df, scores):
    df['intelligibility scores'] = np.zeros(len(df.index))
    
    for i, w1 in enumerate(scores.iloc[:, 0]):
        for j, w2 in enumerate(df.iloc[:, 0]):
            if w1 == w2:
                score = scores.iloc[i, 2]
               #  print(df.iloc[j, -1], score)
                df.iloc[j, -1] = score
    return df

#############################################
# Helper functions for visualizations
#############################################


def compute_levenshtein_heatmap(foreign_alignment, native_alignment, delete_costs=1.0, insert_costs=1.0, subs_costs=None):
    def get_costs(a, b):
        cost = 1
        try:
            if subs_costs is not None:
                cost = subs_costs.loc[a, b]
        except KeyError:
            if a == '-':
                cost = insert_costs
            elif b == '-':
                cost= delete_costs
        return cost
    
    source_align = list(foreign_alignment)
    target_align = list(native_alignment)

   
    # Construct new data frame
    df = pd.DataFrame(np.zeros(shape=(len(source_align), len(target_align))), index=source_align, columns=target_align)
    
    # Fill diagonal entries
    for i in range(len(source_align)):
        sc = df.index[i]
        tc = df.columns[i]
        
        df.iloc[i, i] = get_costs(sc, tc)

    return df


def compute_surprisal_heatmap(foreign_alignment, native_alignment, surprisals=None):
    def get_surprisal(a, b):
        surprisal = surprisals.loc[a, b]
        return surprisal
    
    source_align = list(foreign_alignment)
    target_align = list(native_alignment)
    
    # print('Source alignment: {0}'.format(source_align))
    # print('Target alignment: {0}'.format(target_align))
   
    # Construct new data frame
    df = pd.DataFrame(np.zeros(shape=(len(source_align), len(target_align))), index=source_align, columns=target_align)
    
    # Fill diagonal entries
    for i in range(len(source_align)):
        sc = df.index[i]
        tc = df.columns[i]
        
        df.iloc[i, i] = get_surprisal(sc, tc)

    return df


def multi_column_frame(costs_heatmap, surprisals_heatmaps):
    # Collect data from heatmaps
    costs = np.diag(costs_heatmap)
    costs = np.reshape(costs, (1, len(costs)))
    
    surprisals = np.diag(surprisals_heatmaps)
    surprisals = np.reshape(surprisals, (1, len(surprisals)))
    
    # Reshape data accordingly
    data = np.asarray([costs, surprisals])
    data = np.reshape(data, newshape=(2, -1))

    source_align = costs_heatmap.index
    target_align = costs_heatmap.columns
    
    df = pd.DataFrame(data, index=['costs', 'surprisal'], columns=[source_align, target_align])
    
    return df