import numpy as np

class RealCodecGA_JGG_SPX:
    '''
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）
    初期化
    交叉する個体の選択
    交叉
    残す個体の選択
    '''
    def __init__(self, gene_num, 
                 initial_min, initial_max, 
                 population, 
                 crossover_num, child_num,
                 evaluation_func, 
                 seed=None):
        #乱数シード
        np.random.seed(seed=seed)

        # 遺伝子情報
        self.gene_num = gene_num

        #初期化
        self.initial_min = initial_min
        self.initial_max = initial_max

        # 交叉する個体の選択
        self.population = population
        self.crossed_num = crossover_num

        # 交叉
        self.child_num = child_num

        # 残す個体の選択
        # evaluation function defined as following.
        #  func(genes):
        #    return evaluations
        self.evaluation_func = evaluation_func

        #
        self.genes = None # 候補の遺伝子
        self.evals = None # 候補の遺伝子の評価値
        self.best_gene = None # 現時点で評価値が一番高い遺伝子
        self.best_evaluation = None # 現時点で一番高い評価値

        #
        self.__initialize_genes()

        return

    def __initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)
        
        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx]

        return

    def generation_step(self):
        # 個体数
        pop = len(self.genes)

        # 交叉する個体の選択
        # 交叉する個体のインデックス
        crossed_idx = self.__random_select(pop, self.crossed_num)

        # 交叉
        child_genes = self.__simplex_crossover(self.genes[crossed_idx], self.child_num)

        #残す個体の選択
        survive_genes, survive_evals = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
        #
        self.genes[crossed_idx] = survive_genes
        self.evals[crossed_idx] = survive_evals

        # 最高評価の更新
        if self.best_evaluation > np.min(survive_evals):
            min_idx = np.argmin(survive_evals)
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx]

        return

    def calc_diversity(self):
        dvs = np.average(np.std(self.genes, axis=0))
        return dvs

    @staticmethod
    def __random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def __simplex_crossover(genes, child_num):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # expansion_rate
        expansion_rate = np.sqrt(crss_num + 1.0)

        # Qk計算, Q_(k-1) = e * (P_(k-1) - P_(k)), k=1,2,...,n
        # n = crss_num-1
        diff_genes = np.diff(genes, n=1, axis=0) * (-1.0) * expansion_rate

        # R行列計算
        # [[uniform]*(crsvr_num-1)]*child_num
        rand_coef = np.random.rand(child_num, crss_num-1)
        rand_coef = np.power(rand_coef, 1.0/(np.arange(crss_num-1)+1.0))
        # [u_0*u_1*...*u_n, u_1*...*u_n, ... , u_(n-1)*u_n, u_n
        rand_coef = np.cumprod(rand_coef[:,::-1], axis=1)[:,::-1]

        # Cn計算
        Cn = np.dot(rand_coef, diff_genes)

        # xn計算
        center_gravity = np.average(genes, axis=0)
        xn = center_gravity + expansion_rate * (genes[-1] - center_gravity)

        # C計算
        child_genes = xn + Cn

        return child_genes

    @staticmethod
    def __ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        #
        survive_genes = genes[survive_idx]
        survive_eval = evals[survive_idx]
        return survive_genes, survive_eval

class RealCodecGA_JGG_AREX:
    '''
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）
    初期化
    交叉する個体の選択
    交叉
    残す個体の選択
    '''
    def __init__(self, gene_num, 
                 evaluation_func, 
                 initial_min, initial_max, 
                 population=None, 
                 crossover_num=None, child_num=None, 
                 initial_expantion_rate=None, learning_rate=None, 
                 seed=None):
        '''
        recommended value of parameter are following.
         crossover_num = gene_num + 1
         child_num = 4 * gene_num (~4 * crossover_num)
         initial_expantion_rate = 1.0
         learning_rate = 1/5/gene_num
        '''

        #乱数シード
        np.random.seed(seed=seed)

        # 遺伝子情報
        self.gene_num = gene_num

        # 個体の評価関数
        # evaluation function defined as following.
        #  func(genes):
        #    return evaluations
        self.evaluation_func = evaluation_func

        #遺伝子の初期値範囲
        self.initial_min = initial_min
        self.initial_max = initial_max

        # 全個体数
        # 交叉する個体数
        if (population is not None) and (crossover_num is not None):
            self.population = population
            self.crossed_num = crossover_num
        elif (population is None) and (crossover_num is not None):
            self.crossed_num = crossover_num
            self.population = self.crossed_num
        elif (population is not None) and (crossover_num is None):
            self.population = population
            self.crossed_num = np.minimum(gene_num+1, population)
        elif (population is None) and (crossover_num is None):
            self.crossed_num = gene_num+1
            self.population = self.crossed_num

        # 交叉して生成する子個体数
        self.child_num = child_num if child_num is not None else 4*self.crossed_num

        # 拡張率
        self.expantion_rate = initial_expantion_rate if initial_expantion_rate is not None else 1.0
        # 拡張率の学習率
        self.learning_rate = learning_rate if learning_rate is not None else 1.0/5.0 / gene_num

        #
        self.genes = None # 候補の遺伝子
        self.evals = None # 候補の遺伝子の評価値
        self.best_gene = None # 現時点で評価値が一番高い遺伝子
        self.best_evaluation = None # 現時点で一番高い評価値
        self.last_gene = None # 最終時点での遺伝子
        
        #
        self.__initialize_genes()

        return

    def __initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)
        
        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx].copy()

        return

    def generation_step(self):
        # 個体数
        pop = len(self.genes)

        # 交叉する個体の選択
        # 交叉する個体のインデックス
        crossed_idx = self.__random_select(pop, self.crossed_num)

        # 交叉
        child_genes, rand_mtrx = self.__arex_crossover(self.genes[crossed_idx], self.evals[crossed_idx], self.child_num, self.expantion_rate)

        # 残す個体の選択
        survive_genes, survive_evals, survive_idx = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
        # 親個体と変更
        self.genes[crossed_idx] = survive_genes.copy()
        self.evals[crossed_idx] = survive_evals

        # 拡張率の更新
        self.expantion_rate = self.__update_arex_expantion_rate(self.expantion_rate, self.learning_rate, rand_mtrx[survive_idx], crss_num=self.crossed_num)
        #print(self.expantion_rate)

        min_idx = np.argmin(survive_evals)
        # 最終遺伝子の更新
        self.last_gene = survive_genes[min_idx].copy()
        # 生き残ったうちの最高評価
        best_survive_evals = survive_evals[min_idx]
        # 最高評価の更新
        if self.best_evaluation > survive_evals[min_idx]:
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx].copy()
        
        return best_survive_evals

    def calc_diversity(self):
        dvs = np.average(np.std(self.genes, axis=0))
        return dvs

    @staticmethod
    def __random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def __arex_crossover(genes, evals, child_num, expantion_rate):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # 評価が高い順に並べたインデックス
        sorted_idx = np.argsort(evals)

        # 荷重重心
        w = 2.0 * (crss_num + 1.0 - np.arange(1, crss_num+1)) / (crss_num * (crss_num + 1.0))
        wG = np.dot(w[np.newaxis,:], genes[sorted_idx])

        # 重心
        G = np.average(genes, axis=0)

        # 乱数
        rnd_mtrx = np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num))
        
        # 子個体
        child_genes = wG + expantion_rate * np.dot(rnd_mtrx, genes - G)

        return child_genes, rnd_mtrx

    @staticmethod
    def __update_arex_expantion_rate(expantion_rate, learning_rate, survive_rand_mtrx, crss_num=None):
        survive_num = len(survive_rand_mtrx)
        #
        crss_num_ = crss_num if crss_num is not None else survive_num
        #
        ave_r = np.average(survive_rand_mtrx, axis=0)
        L_cdp = expantion_rate**2 * (survive_num - 1) * (np.sum(np.square(ave_r)) - (np.sum(ave_r))**2 / survive_num)
        L_ave = expantion_rate**2 * (1/(crss_num_ - 1)) * (survive_num - 1)**2 / survive_num
        #
        new_expantion_rate = np.maximum(1.0, expantion_rate * np.sqrt((1.0 - learning_rate) + learning_rate * L_cdp / L_ave))

        return new_expantion_rate

    @staticmethod
    def __ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        #
        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]
        return survive_genes, survive_eval, survive_idx

class Sample:
    @staticmethod
    def sample_RealCodecGA_JGG_SPX():
        # define evaluation func
        def minus_l2(ndar):
            ev = - np.sum(np.square(ndar), axis=1)
            return ev
        # rcga class
        rcga = RealCodecGA_JGG_SPX(gene_num=2, 
                                   initial_min=-1, initial_max=1, 
                                   population=10, 
                                   crossover_num=3, child_num=10,
                                   evaluation_func=minus_l2, 
                                   seed=1000)

        # optimize
        for i in range(100):
            rcga.generation_step()
            print('{0}'.format(i+1))
            print(' best evals : {0}'.format(rcga.best_evaluation))
            print(' best genes : {0}'.format(rcga.best_gene))
        return
    @staticmethod
    def sample_RealCodecGA_JGG_AREX():
        # define evaluation func
        def minus_l2(ndar):
            ev = - np.sum(np.square(ndar), axis=1)
            return ev
        # rcga class
        rcga = RealCodecGA_JGG_AREX(gene_num=2, 
                                    evaluation_func=minus_l2, 
                                    initial_min=-1, initial_max=1, 
                                    population=10, 
                                    crossover_num=3, child_num=6, 
                                    initial_expantion_rate=1.0, learning_rate=0.05, 
                                    seed=1000)
        # optimize
        for i in range(100):
            rcga.generation_step()
            print('{0}'.format(i+1))
            print(' best evals : {0}'.format(rcga.best_evaluation))
            print(' best genes : {0}'.format(rcga.best_gene))
        return

if __name__ == '__main__':
    #test()
    import verification
    
    #verification.test_Minimize_L2_1()
    verification.test_LinearLeastSquares_withRCGA()