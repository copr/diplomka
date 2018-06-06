
class ProposalDistribution:
    '''
    trida reprezentujici navrhovou distribuci

    funkce pdf by mela brat predchozi sample a novy sample a  vracet
      pravdepodobnost prechodu z predchoziho do noveho
    funkce rvs by mela brat 'predchozi sample' a vracet novy sample
    '''
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs
