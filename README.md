# TabNet for fastai
> This is an adaptation of TabNet (Attention-based network for tabular data) for fastai (>=2.0) library. The original paper https://arxiv.org/pdf/1908.07442.pdf. The implementation is taken from here https://github.com/dreamquark-ai/tabnet


## Install

`pip install fast_tabnet`

## How to use

`model = TabNetModel(emb_szs, n_cont, out_sz, embed_p=0., y_range=None, 
                     n_d=8, n_a=8,
                     n_steps=3, gamma=1.3, 
                     n_independent=2, n_shared=2, epsilon=1e-15,
                     virtual_batch_size=128, momentum=0.02)`

Parameters `emb_szs, n_cont, out_sz, embed_p, y_range` are the same as for fastai TabularModel.

- n_d : int
    Dimension of the prediction  layer (usually between 4 and 64)
- n_a : int
    Dimension of the attention  layer (usually between 4 and 64)
- n_steps: int
    Number of sucessive steps in the newtork (usually betwenn 3 and 10)
- gamma : float
    Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
- momentum : float
    Float value between 0 and 1 which will be used for momentum in all batch norm
- n_independent : int
    Number of independent GLU layer in each GLU block (default 2)
- n_shared : int
    Number of independent GLU layer in each GLU block (default 2)
- epsilon: float
    Avoid log(0), this should be kept very low


## Example

Below is an example from fastai library, but the model in use is TabNet

```
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *
```

```
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



```
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 
             'relationship', 'race', 'native-country', 'sex']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
```

```
to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="salary", y_block = CategoryBlock(), splits=splits)
```

```
dls = to.dataloaders()
```

```
dls.valid.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>native-country</th>
      <th>sex</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Married-civ-spouse</td>
      <td>Adm-clerical</td>
      <td>Wife</td>
      <td>Black</td>
      <td>United-States</td>
      <td>Female</td>
      <td>False</td>
      <td>27.000000</td>
      <td>200226.999829</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>54.000000</td>
      <td>171924.000299</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>United-States</td>
      <td>Female</td>
      <td>False</td>
      <td>22.000000</td>
      <td>268144.999557</td>
      <td>13.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Self-emp-not-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>51.000000</td>
      <td>95435.000919</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>State-gov</td>
      <td>7th-8th</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>47.000000</td>
      <td>103743.000887</td>
      <td>4.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>63.000001</td>
      <td>187919.000042</td>
      <td>7.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>?</td>
      <td>Some-college</td>
      <td>Separated</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>30.000000</td>
      <td>97280.999859</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Self-emp-not-inc</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>47.000000</td>
      <td>107230.998424</td>
      <td>12.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>18.999999</td>
      <td>459247.992218</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>Masters</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>United-States</td>
      <td>Male</td>
      <td>False</td>
      <td>39.000000</td>
      <td>87555.998074</td>
      <td>14.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


```
to_tst = to.new(df_test)
to_tst.process()
to_tst.all_cols.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>native-country</th>
      <th>sex</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10000</th>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>31</td>
      <td>2</td>
      <td>1</td>
      <td>0.463422</td>
      <td>1.343164</td>
      <td>1.170645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10001</th>
      <td>5</td>
      <td>12</td>
      <td>3</td>
      <td>15</td>
      <td>1</td>
      <td>4</td>
      <td>40</td>
      <td>2</td>
      <td>1</td>
      <td>-0.937841</td>
      <td>1.255783</td>
      <td>-0.430442</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>5</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>1.053427</td>
      <td>0.148568</td>
      <td>-1.230986</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>5</td>
      <td>12</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>0.537172</td>
      <td>-0.286319</td>
      <td>-0.430442</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>40</td>
      <td>2</td>
      <td>1</td>
      <td>0.758424</td>
      <td>1.446120</td>
      <td>0.370101</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```
emb_szs = get_emb_sz(to); print(emb_szs)
```

    [(10, 6), (17, 8), (8, 5), (16, 8), (7, 5), (6, 4), (43, 13), (3, 3), (3, 3)]


That's the use of the model

```
model = TabNetModel(emb_szs, len(to.cont_names), dls.c, n_d=8, n_a=32, n_steps=1); 
```

```
opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dls, model, CrossEntropyLossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
```

```
learn.lr_find()
```








    SuggestedLRs(lr_min=0.15848932266235352, lr_steep=0.10000000149011612)




![png](docs/images/output_19_2.png)


```
learn.fit_one_cycle(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.493278</td>
      <td>0.468720</td>
      <td>0.752500</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.412112</td>
      <td>0.401659</td>
      <td>0.823000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.398074</td>
      <td>0.383860</td>
      <td>0.814500</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.390411</td>
      <td>0.398729</td>
      <td>0.819500</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.374312</td>
      <td>0.363800</td>
      <td>0.833000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.367166</td>
      <td>0.372629</td>
      <td>0.827000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.357769</td>
      <td>0.358854</td>
      <td>0.835000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.352027</td>
      <td>0.361226</td>
      <td>0.835500</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.342584</td>
      <td>0.360935</td>
      <td>0.835500</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.334791</td>
      <td>0.361685</td>
      <td>0.836000</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


## Hyperparameter search with Bayesian Optimization

I like to tune hyperparameters for tabular models with Bayesian Optimization. You can optimize directly your metric using this approach if the metric is sensitive enough (in our example it is not and we use validation loss instead). Also, you should create the second validation set, because you will use the first as a training set for Bayesian Optimization. 


You may need to install the optimizer `pip install bayesian-optimization`

```
from functools import lru_cache
```

```
# The function we'll optimize
@lru_cache(1000)
def get_accuracy(n_d:Int, n_a:Int, n_steps:Int):
    model = TabNetModel(emb_szs, len(to.cont_names), dls.c, n_d=n_d, n_a=n_a, n_steps=n_steps);
    learn = Learner(dls, model, CrossEntropyLossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
    learn.fit_one_cycle(5)
    return float(learn.validate(dl=learn.dls.valid)[1])
```

This implementation of Bayesian Optimization doesn't work naturally with descreet values. That's why we use wrapper with `lru_cache`.

```
def fit_accuracy(pow_n_d, pow_n_a, pow_n_steps):
    n_d, n_a, n_steps = map(lambda x: 2**int(x), (pow_n_d, pow_n_a, pow_n_steps))
    return get_accuracy(n_d, n_a, n_steps)
```

```
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'pow_n_d': (0, 8), 'pow_n_a': (0, 8), 'pow_n_steps': (0, 4)}

optimizer = BayesianOptimization(
    f=fit_accuracy,
    pbounds=pbounds,
)
```

```
optimizer.maximize(
    init_points=15,
    n_iter=100,
)
```

    |   iter    |  target   |  pow_n_a  |  pow_n_d  | pow_n_... |
    -------------------------------------------------------------
    | [0m 1       [0m | [0m 0.815   [0m | [0m 1.39    [0m | [0m 3.331   [0m | [0m 2.779   [0m |
    | [95m 2       [0m | [95m 0.818   [0m | [95m 1.328   [0m | [95m 6.287   [0m | [95m 2.564   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.492287</td>
      <td>0.457265</td>
      <td>0.765500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.429238</td>
      <td>0.408366</td>
      <td>0.816500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.420226</td>
      <td>0.401817</td>
      <td>0.820500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.408870</td>
      <td>0.396778</td>
      <td>0.806500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.397386</td>
      <td>0.393232</td>
      <td>0.817000</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>






    | [0m 3       [0m | [0m 0.817   [0m | [0m 3.747   [0m | [0m 0.1692  [0m | [0m 1.442   [0m |




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='3' class='' max='5', style='width:300px; height:20px; vertical-align: middle;'></progress>
      60.00% [3/5 00:12<00:08]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.672914</td>
      <td>0.509420</td>
      <td>0.764500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.469479</td>
      <td>0.451038</td>
      <td>0.764500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.426840</td>
      <td>0.418067</td>
      <td>0.770000</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='40' class='' max='125', style='width:300px; height:20px; vertical-align: middle;'></progress>
      32.00% [40/125 00:01<00:02 0.4247]
    </div>



```
optimizer.max['target']
```




    0.8349999785423279



```
{key: 2**int(value)
  for key, value in optimizer.max['params'].items()}
```




    {'pow_n_a': 32, 'pow_n_d': 128, 'pow_n_steps': 1}


