import numpy as np



class PO:
    def __init__(self):
        self.fun = None
        self.lambdamax = 0
        self.n = 0
        self.tmax = 0
        self.d = 0

    def ElectionCampaign(self, fun, d, pji, pjitminus1, pstari, cstarj):
        if (fun(pji) <= fun(pjitminus1)):
            for k in range(d):
                mstar = pstari[k]
                r = np.random.rand()
                # Implementation of eq9
                if ((pjitminus1[k] <= pji[k] <= mstar) or (pjitminus1[k] >= pji[k] >= mstar)):
                    pji[k] = mstar + r * (mstar - pji[k])
                elif ((pjitminus1[k] <= mstar <= pji[k]) or (pjitminus1[k] >= mstar >= pji[k])):
                    pji[k] = mstar = (2 * r - 1) * abs(mstar - pji[k])
                elif ((mstar <= pjitminus1[k] <= pji[k]) or (mstar >= pjitminus1[k] >= pji[k])):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pjitminus1[k])
                mstar = cstarj[k]
                r = np.random.rand()
                # This is just a copy paste of what was above, since eq 9 already implemented
                if ((pjitminus1[k] <= pji[k] <= mstar) or (pjitminus1[k] >= pji[k] >= mstar)):
                    pji[k] = mstar + r * (mstar - pji[k])
                elif ((pjitminus1[k] <= mstar <= pji[k]) or (pjitminus1[k] >= mstar >= pji[k])):
                    pji[k] = mstar = (2 * r - 1) * abs(mstar - pji[k])
                elif ((mstar <= pjitminus1[k] <= pji[k]) or (mstar >= pjitminus1[k] >= pji[k])):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pjitminus1[k])
        else:
            for k in range(d):
                mstar = pstari[k]
                r = np.random.rand()
                # Implementation of eq 10
                if ((pjitminus1[k] <= pji[k] <= mstar) or (pjitminus1[k] >= pji[k] >= mstar)):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pji[k])
                elif ((pjitminus1[k] <= mstar <= pji[k]) or (pjitminus1[k] >= mstar >= pji[k])):
                    pji[k] = pjitminus1[k] + r * (pji[k] - pjitminus1[k])
                elif ((mstar <= pjitminus1[k] <= pji[k]) or (mstar >= pjitminus1[k] >= pji[k])):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pjitminus1[k])
                mstar = cstarj[k]
                r = np.random.rand()
                # Copy paste of what was above, since eq 10 already implemented
                if ((pjitminus1[k] <= pji[k] <= mstar) or (pjitminus1[k] >= pji[k] >= mstar)):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pji[k])
                elif ((pjitminus1[k] <= mstar <= pji[k]) or (pjitminus1[k] >= mstar >= pji[k])):
                    pji[k] = pjitminus1[k] + r * (pji[k] - pjitminus1[k])
                elif ((mstar <= pjitminus1[k] <= pji[k]) or (mstar >= pjitminus1[k] >= pji[k])):
                    pji[k] = mstar + (2 * r - 1) * abs(mstar - pjitminus1[k])
        return pji

    def PartySwitching(self, parties, population, Lambda, fitnesst, n):
        for i in range(len(parties)):
            # got the ith party
            for j in range(len(parties[i])):
                # for each member of the party
                sp = np.random.rand()
                if (sp < Lambda):
                    r = np.random.randint(0, n - 1)  # the random new party (from 0 to n-1 as it how python indexes)
                    q = np.argmax(fitnesst[parties[r]])  # do we have to recompute here? or time consuming
                    old = parties[i][j].copy()
                    new = parties[r][q].copy()
                    # print('before')
                    # print(parties[i])
                    # print(parties[r])
                    # we swap :
                    parties[i][j] = new
                    parties[r][q] = old
                    # print('after')
                    # print(parties[i])
                    # print(parties[r])
                    # Determine q with eq11
                    # swap pqr and pji
        return parties

    def ParliamentaryAffairs(self,constituencyLeaders, population, n, fun):
        for j in range(n):
            r = np.random.randint(0, n - 1)
            # to make sure that r != j
            while r == j:
                r = np.random.randint(0, n - 1)

            a = np.random.rand()
            c_j = population[int(constituencyLeaders[j])]
            c_r = population[int(constituencyLeaders[r])]
            c_new = c_r + (2 * a - 1) * np.linalg.norm(c_r - c_j)

            print(c_new)
            fitness_new = fun(c_new)
            fitness_j = fun(c_j)
            if fitness_new <= fitness_j:
                population[int(constituencyLeaders[j])] = c_new

        return 0

    def PoliticalOptimizer(self, fun, n, lambdamax, tmax, d):
        '''
        Parameters:
        fun -> the function we want to optimize
        n -> the number of contituencies, political parties and party members
        lambdamax -> upper limit of the party switching rate
        tmax -> number of iterations

        d -> number of dimensions, we don't want this to be a parameter later I think

        Out: the final population

        '''
        self.fun = fun
        self.n = n
        self.lambdamax = lambdamax
        self.tmax = tmax
        self.d = d

        ################
        # Initialization: sth arbitrary now to replicate the paper, making 9 2D individuals
        # n = 3 parties, d = 2 dimensions
        ################

        populationt = np.zeros((self.n ** 2, self.d))
        for i in range(self.n ** 2):
            populationt[i, :] = np.random.normal(0, 1, self.d)
        print(populationt)
        ###############
        # Initializing parties
        # We have n parties and constituencies in the political system, we initialize them here
        # party 1 consists of the first n individuals, party 2 of the following n etc.
        # we store the indices in parties, so that we can always use the indices to refer to individuals in the population
        ###############
        parties = constituencies = np.array([range(self.n * i, (self.n * i + self.n)) for i in range(self.n)])
        print(f'The parties: {parties}')

        ###############
        # Initialize the fitness of all individuals
        ###############
        fitnesst = np.array([fun(x) for x in populationt])
        print(f'Their fitness: {fitnesst}')
        #################
        # Computing the party leaders
        #################

        # First we look at the first party, then the second, then the third,...

        n_party = 0
        party_leaders = np.zeros(self.n)
        for party in parties:
            party_leaders[n_party] = n_party * self.n + np.argmin(fitnesst[party])
            # 0 * 3 + 0,1 or 2. Then 1*3 + 0,1,2
            n_party += 1

        print(f'The selected party leaders: {party_leaders}')

        ################
        # Computing the constituency leaders, 1 member for each party competes against each other
        # So in our toy example it is individual 0 vs. 3 vs. 6
        # In general it is 0 * n, 1 * n ...
        ################

        # first we make the constituencies
        constituencies = []

        for const in range(self.n):
            constituency = []
            for party in parties:
                constituency.append(party[const])
            constituencies.append(constituency)
        # then we select the leader
        n_constituency = 0
        constituency_leaders = np.zeros(self.n)
        for constituency in constituencies:
            # For constituency 0: 0, 3 or 6
            # For constituency 1: 1, 4 or 7
            constituency_leaders[n_constituency] = n_constituency + 3 * np.argmin(fitnesst[constituency])
            n_constituency += 1

        print(f'The constituency leaders: {constituency_leaders}')

        ################
        # Further initialisations
        ################
        t = 1
        populationtmin1 = populationt.copy()
        fitnesstmin1 = fitnesst.copy()
        Lambda = lambdamax
        print(f'index of p00 is: {parties[0][0]}')
        print(f'individual p00 is: {populationt[parties[0, 0]]}')
        ################
        # Starting while loop
        ################
        while (t <= tmax):
            populationtemp = populationt.copy()

            fitnesstemp = fitnesst.copy()
            for i in range(len(parties)):
                for j in range(len(party)):
                    # partyleaders and constituency leaders give the index of the party leader
                    # we can plug this into the population to get the party leader
                    populationt[parties[i, j]] = self.ElectionCampaign(fun, d, pji=populationt[parties[i, j]].copy(),
                                                                  pjitminus1=populationtmin1[parties[i, j]].copy(),
                                                                  pstari=populationt[int(party_leaders[i])].copy(),
                                                                  cstarj=populationt[
                                                                      int(constituency_leaders[j])].copy())

            parties = self.PartySwitching(parties, populationt, Lambda, fitnesst, n)

            # fitness of each party member pji
            fitnesstemp = np.array([fun(x) for x in populationtemp])

            # finding party leaders
            n_party = 0
            party_leaders = np.zeros(self.n)
            for party in parties:
                party_leaders[n_party] = n_party * self.n + np.argmin(fitnesstemp[party])
                n_party += 1

            # then we select the leader
            n_constituency = 0
            constituency_leaders_loop = np.zeros(self.n)
            for constituency in constituencies:
                constituency_leaders_loop[n_constituency] = n_constituency + 3 * np.argmin(fitnesst[constituency])
                n_constituency += 1

            self.ParliamentaryAffairs(constituency_leaders_loop, populationt, self.n, fun);

            populationtmin1 = populationtemp
            fitnesstmin1 = fitnesstemp
            Lambda = Lambda - float(lambdamax) / tmax
            t += 1

po = PO()
po.PoliticalOptimizer(lambda x: x[0]**2 + x[1]**2,3,2,2,2)
