import numpy as np

class Move:
    '''
    predstavuje prechod z k do k' a zpet
    '''

    def __init__(self,
                 k1,
                 k2,
                 k1_to_k2,
                 k2_to_k1,
                 transform1to2,
                 transform2to1,
                 jacobian1to2,
                 jacobian2to1,
                 ugenerator1to2,
                 ugenerator2to1,
                 usize1,
                 usize2):
        '''
        k1 - prvni stav
        k2 - druhy stav
        k1_to_k2 - pravdepodobnost prechodu z k1 do k2
        k2_to_k1 - pravdepodobnost prechodu z k2 do k1
        transform1to2 - pretransformuje vzorek na vzorek s jinou dimenzi
                        vraci novy vzorek a mozne nahodne cisla, ktere
                        by byly vegenerovane pro zpetnou transformaci
        transform2to1 - ---||---
        ugenerator1to2 - generator pomocnych nahodnych velicin pro prechod
                         z k1 do k2
        ugenerator2to1 - ---||---
        usize1 - delka vektoru nahodnych velicin ktery se vygeneruje pro prechod ze 
                 stavu 1 do stavu 2
        usize2 - ---||---
        '''
        self.k1 = k1
        self.k2 = k2
        self.k1_to_k2 = k1_to_k2
        self.k2_to_k1 = k2_to_k1
        self.transform1to2 = transform1to2
        self.transform2to1 = transform2to1
        self.jacobian1to2 = jacobian1to2
        self.jacobian2to1 = jacobian2to1
        self.ugenerator1to2 = ugenerator1to2
        self.ugenerator2to1 = ugenerator2to1
        self.usize1 = usize1
        self.usize2 = usize2

    def can_move(self, k):
        '''
        urci jestli muzu pouzit tento prechod z daneho stavu
        '''
        if k == self.k1:
            return self.k1_to_k2
        elif k == self.k2:
            return self.k2_to_k1
        else:
            return 0

    def probability_of_this_move(self, x):
        '''
        pravdepodobnost pouziti tohole typu pohybu z daneho
        stavu. zatim pouzivame jen dany stav k, ale v algoritmu
        je mozne i pouzit pro vypocet pravdepodobnosti soucasny
        vzorek. takze ted ty prechodove pravdepodobnosti jsou jenom
        konstanty, ale obecne by to mohla byt i funkce
        '''
        k, theta = x
        if self.k1 == k:
            return self.k1_to_k2
        elif self.k2 == k:
            return self.k2_to_k1
        else:
            raise RuntimeError("This should never happen")

    def _transform(self, k, newk, theta, generator, transform, newu_size):
        if generator is not None:
            u = generator.rvs()
            ext_theta = np.append(theta, u)
        else:
            u = None
            ext_theta = theta
        newtheta = transform(ext_theta)

        if newu_size is not 0:
            newu = newtheta[-newu_size:]
            newtheta = newtheta[:-newu_size]
        else:
            newu = None
        newx = (newk, newtheta)
        det_jacobian = self.get_jacobian((k, ext_theta))
        return (newx, u, newu, det_jacobian)

    def transform(self, x):
        # tady by se rovnou mohl vracet i ten jacobian at se s tim neseru venku
        k, theta = x
        if self.k1 == k:
            return self._transform(k,
                                   self.k2,
                                   theta,
                                   self.ugenerator1to2,
                                   self.transform1to2,
                                   self.usize2)
        elif self.k2 == k:
            return self._transform(k,
                                   self.k1,
                                   theta,
                                   self.ugenerator2to1,
                                   self.transform2to1,
                                   self.usize1)
        else:
            raise RuntimeError("This shoould never happen")

    # def transform(self, x):
    #     k, theta = x
    #     if self.k1 == k:
    #         if self.ugenerator1to2 is not None:
    #             u = self.ugenerator1to2.rvs()
    #             ext_theta = np.append(theta, u)
    #         else:
    #             u = None
    #             ext_theta = theta
    #         newtheta = self.transform1to2(ext_theta)
    #         # hack
    #         newx = (self.k2, newtheta)
    #         return (newx, u, None)
    #     elif self.k2 == k:
    #         if self.ugenerator2to1 is not None:
    #             u = self.generator2to1.rvs()
    #             ext_theta = np.append(theta, u)
    #         else:
    #             u = None
    #             ext_theta = theta
    #         newtheta = self.transform2to1(ext_theta)
    #         newu = newtheta[-2:]
    #         newx = (self.k1, newtheta[:-2])
    #         return (newx, None, newu)
    #     else:
    #         raise RuntimeError("This should never happen")

    def probability_of_help_rvs(self, x, u):
        k, theta = x
        if self.k1 == k:
            if self.ugenerator1to2 is None:
                return 1
            return self.ugenerator1to2.pdf(u)
        elif self.k2 == k:
            if self.ugenerator2to1 is None:
                return 1
            return self.ugenerator2to1.pdf(u)
        else:
            raise RuntimeError("This should never happen")

    def get_jacobian(self, x):
        k, theta = x
        if self.k1 == k:
            mat = self.jacobian1to2(theta)
            return np.linalg.det(mat)
        elif self.k2 == k:
            mat = self.jacobian2to1(theta)
            return np.linalg.det(mat)
        else:
            raise RuntimeError("This should never happen")
