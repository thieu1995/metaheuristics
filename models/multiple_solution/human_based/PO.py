import numpy as np

class PO:
    def __init__(self, po_params):
        '''
        Parameters:
        fun -> the function we want to optimize
        n -> the number of contituencies, political parties and party members
        lambdamax -> upper limit of the party switching rate
        tmax -> number of iterations

        d -> number of dimensions, we don't want this to be a parameter later I think

        Out: the final population

        '''
        self.fun = po_params['fun']
        self.lambdamax = po_params['lambdamax']
        self.n = po_params['n']
        self.tmax = po_params['tmax']
        self.lambdamax = po_params['lambdamax']
        self.d = po_params['d']

        
    def ElectionCampaign(self, leader_position, leader_minus_1, pstari, cstarj):
        if (self.fun(leader_position) <= self.fun(leader_minus_1)):
            for k in range(self.d):
                mstar = pstari[k]
                r = np.random.rand()
                # Implementation of eq9
                if ((leader_minus_1[k] <= leader_position[k] <= mstar) or (leader_minus_1[k] >= leader_position[k] >= mstar)):
                    leader_position[k] = mstar + r * (mstar - leader_position[k])
                elif ((leader_minus_1[k] <= mstar <= leader_position[k]) or (leader_minus_1[k] >= mstar >= leader_position[k])):
                    leader_position[k] = mstar = (2 * r - 1) * abs(mstar - leader_position[k])
                elif ((mstar <= leader_minus_1[k] <= leader_position[k]) or (mstar >= leader_minus_1[k] >= leader_position[k])):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_minus_1[k])
                mstar = cstarj[k]
                r = np.random.rand()
                # This is just a copy paste of what was above, since eq 9 already implemented
                if ((leader_minus_1[k] <= leader_position[k] <= mstar) or (leader_minus_1[k] >= leader_position[k] >= mstar)):
                    leader_position[k] = mstar + r * (mstar - leader_position[k])
                elif ((leader_minus_1[k] <= mstar <= leader_position[k]) or (leader_minus_1[k] >= mstar >= leader_position[k])):
                    leader_position[k] = mstar = (2 * r - 1) * abs(mstar - leader_position[k])
                elif ((mstar <= leader_minus_1[k] <= leader_position[k]) or (mstar >= leader_minus_1[k] >= leader_position[k])):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_minus_1[k])
        else:
            for k in range(self.d):
                mstar = pstari[k]
                r = np.random.rand()
                # Implementation of eq 10
                if ((leader_minus_1[k] <= leader_position[k] <= mstar) or (leader_minus_1[k] >= leader_position[k] >= mstar)):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_position[k])
                elif ((leader_minus_1[k] <= mstar <= leader_position[k]) or (leader_minus_1[k] >= mstar >= leader_position[k])):
                    leader_position[k] = leader_minus_1[k] + r * (leader_position[k] - leader_minus_1[k])
                elif ((mstar <= leader_minus_1[k] <= leader_position[k]) or (mstar >= leader_minus_1[k] >= leader_position[k])):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_minus_1[k])
                mstar = cstarj[k]
                r = np.random.rand()
                # Copy paste of what was above, since eq 10 already implemented
                if ((leader_minus_1[k] <= leader_position[k] <= mstar) or (leader_minus_1[k] >= leader_position[k] >= mstar)):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_position[k])
                elif ((leader_minus_1[k] <= mstar <= leader_position[k]) or (leader_minus_1[k] >= mstar >= leader_position[k])):
                    leader_position[k] = leader_minus_1[k] + r * (leader_position[k] - leader_minus_1[k])
                elif ((mstar <= leader_minus_1[k] <= leader_position[k]) or (mstar >= leader_minus_1[k] >= leader_position[k])):
                    leader_position[k] = mstar + (2 * r - 1) * abs(mstar - leader_minus_1[k])
        return leader_position

    def PartySwitching(self, parties, population, Lambda, fitnesst):
        for i in range(len(parties)):
            # got the ith party
            for j in range(len(parties[i])):
                # for each member of the party
                sp = np.random.rand()
                if (sp < Lambda):
                    r = np.random.randint(0, self.n - 1)  # the random new party (from 0 to n-1 as it how python indexes)
                    q = np.argmax(fitnesst[parties[r]]) 
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

    def ParliamentaryAffairs(self, constituencyLeaders, population):
        for j in range(self.n):
            r = np.random.randint(0, self.n - 1)
            # to make sure that r != j
            while r == j:
                r = np.random.randint(0, self.n - 1)

            a = np.random.rand()
            c_j = population[int(constituencyLeaders[j])]
            c_r = population[int(constituencyLeaders[r])]
            c_new = c_r + (2 * a - 1) * np.linalg.norm(c_r - c_j)

            #print(c_new)
            fitness_new = self.fun(c_new)
            fitness_j = self.fun(c_j)
            if fitness_new <= fitness_j:
                population[int(constituencyLeaders[j])] = c_new

        return 0

    def _train__(self):
       
        ################
        # Initialization: sth arbitrary now to replicate the paper, making 9 2D individuals
        # n = 3 parties, d = 2 dimensions
        ################

        populationt = np.zeros((self.n ** 2, self.d))
        for i in range(self.n ** 2):
            populationt[i, :] = np.random.normal(0, 1, self.d)
        #print(populationt)
        ###############
        # Initializing parties
        # We have n parties and constituencies in the political system, we initialize them here
        # party 1 consists of the first n individuals, party 2 of the following n etc.
        # we store the indices in parties, so that we can always use the indices to refer to individuals in the population
        ###############
        parties = constituencies = np.array([range(self.n * i, (self.n * i + self.n)) for i in range(self.n)])
        #print(f'The parties: {parties}')

        ###############
        # Initialize the fitness of all individuals
        ###############
        fitnesst = np.array([self.fun(x) for x in populationt])
        #print(f'Their fitness: {fitnesst}')
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

        #print(f'The selected party leaders: {party_leaders}')

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

        #print(f'The constituency leaders: {constituency_leaders}')

        ################
        # Further initialisations
        ################
        t = 1
        populationtmin1 = populationt.copy()
        fitnesstmin1 = fitnesst.copy()
        Lambda = self.lambdamax
        #print(f'index of p00 is: {parties[0][0]}')
        #print(f'individual p00 is: {populationt[parties[0, 0]]}')
        ################
        # Starting while loop
        ################
        self.df = pd.DataFrame()
        while (t <= self.tmax):
            populationtemp = populationt.copy()

            fitnesstemp = fitnesst.copy()
            for i in range(len(parties)):
                for j in range(len(party)):
                    # partyleaders and constituency leaders give the index of the party leader
                    # we can plug this into the population to get the party leader
                    populationt[parties[i, j]] = self.ElectionCampaign(pji=populationt[parties[i, j]].copy(),
                                                                  pjitminus1=populationtmin1[parties[i, j]].copy(),
                                                                  pstari=populationt[int(party_leaders[i])].copy(),
                                                                  cstarj=populationt[
                                                                      int(constituency_leaders[j])].copy())

            parties = self.PartySwitching(parties, populationt, Lambda, fitnesst)

            # fitness of each party member pji
            fitnesstemp = np.array([self.fun(x) for x in populationtemp])

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

            self.ParliamentaryAffairs(constituency_leaders_loop, populationt);

            self.populationtmin1 = populationtemp
            self.fitnesstmin1 = fitnesstemp
            self.df[str(t)] = list(self.fitnesstmin1)
            Lambda = Lambda - float(self.lambdamax) / self.tmax
            t += 1
        return self.populationtmin1[np.argmin(self.fitnesstmin1)], np.min(self.fitnesstmin1)
