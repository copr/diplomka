class ProposalDistribution2:
    '''
    trida reprezentujici navrhovou distribuci

    funkce pdf by mela brat predchozi sample a novy sample a  vracet
      pravdepodobnost prechodu z predchoziho do noveho
    funkce rvs by mela brat 'predchozi sample' a vracet novy sample
    '''
    def __init__(self, rv):
        self.rv = rv

    def pdf(self, x, y):
        self.rv.mean = x
        return self.rv.pdf(y)

    def rvs(self, x):
        self.rv.mean = x
        return self.rv.rvs()
