import torch


class SequenceModel:
    
    def __init__(self, 
                 seq_length, 
                 num_words,
                 num_concepts, 
                 num_seq_per_class, 
                 num_classes):
        """
        Attributes:
        
        A: matrix organizing the words into concepts. 
           For example, if num_words = 12 and num_concepts = 3 then A will be:
           
                 [ 0   1   2   3 ]     <-- concept 0 
           A=    [ 4   5   6   7 ]     <-- concept 1
                 [ 8   9   10  11]     <-- concept 2
                 
        Remark: there is no need to choose the partition at random since the neural net
                is invariant with respect to relabelling the words
                 
            
        sequences_of_concepts: LongTensor of shape (num_classes*num_seq_per_class , seq_length)
                               with entries in {0,...,num_concepts-1}       
                              
                               
        category:  LongTensor of shape shape(num_classes*num_seq_per_class)
                   with entries in {0,...,num_classes-1}
                   This is the vector that indicates to which category each sequence of concept belongs
                   If num_classes = 3, and num_seq_per_class = 5, this vector is
                   [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
                   In otherwords, the first num_seq_per_class sequences of concepts belong to category 0
                   The next num_seq_per_class sequences of concepts belong to category 1
                   etc...
        """
        
        assert num_words%num_concepts==0,'num_words must be div by num_concepts'
        self.concept_size = num_words // num_concepts
        
        self.A = torch.arange(num_words).view(num_concepts, self.concept_size) 
        
        self.sequences_of_concepts = torch.randint(0,num_concepts, 
                                      size = (num_classes*num_seq_per_class,seq_length))
        
        self.category = torch.arange(num_classes).repeat_interleave(num_seq_per_class)
        
        
    def generate_k_sentences_per_seq_of_concept(self,k):
        """
        X: LongTensor of shape (num_classes*num_seq_per_class*k, seq_length ) 
           with entries in {0,...,num_words-1}
           The first k rows of X are sequences randomly generated by the the first seq of concepts
           The next k rows of X are sequences randomly generated by the the second seq of concepts
           etc...
            
        labels: LongTensor of shape (num_classes*num_subclasses*k,) 
                with entries in {0,...,num_classes-1}
        """
        
        # concepts corresponds to rows of the matrix A
        # So given a sequence of concepts I = [i_0, i_1, ..., i_{L-1}],
        # in order to sample a data point, we need a sequence
        #                                 J = [j_0, j_1, ..., j_{L-1}]
        # where the j_l are randomly sampled in {0,...,concept_size - 1}
        # Then A[I,J] is a random sentence generated by the sequence of concept I
        
        # In the code below, I represents multiple sequences of concepts:
        #
        #         [i_00, i_01, ..., i_{0,L-1}]   <-- sequence 0
        #  I  =   [i_10, i_11, ..., i_{1,L-1}]   <-- sequence 1
        #         [i_20, i_21, ..., i_{2,L-1}]   <-- sequence 2
        #
        # and so does J
        
        I = self.sequences_of_concepts.repeat_interleave(k,dim=0)
        J = torch.randint(0,self.concept_size , size = I.shape)
        X = self.A[I,J]
        labels =  self.category.repeat_interleave(k,dim=0)
        return X, labels       