# Document for signallib
##  1. Intro to SignalLib
##  2. Usage manual
##  3. Factor profile
[Example: signal name]
- brief (or ref) : one line description or paper name or url  
- subclass: all return signals
- param: special parameters input by settings['Params']
0. Labels 
   1. TicksABReturn
      - brief: 快照預期收益
      - subclass: [TickAskReturn, TickBidReturn]  
      - param: r: 平移窗口
1. BarMomentum
   1. AverageAmountReturn
      - brief: 收益与平均单笔成交额乘积，抓取大單多空博弈的結果
      - subclass: [收益与平均单笔成交额乘积, 收益与 log(平均单笔成交额)乘积]  
      - param: r: 回看窗口
   2. SmartMoney(停用)
      - brief: 采用指标 S 来衡量每一个分钟交易的“聪明”程度，按照指标 S 从大到小进行排序，将成交量累积占比前 20% 视为聪明钱的交易。
      - subclass: 
      - param: r: w1:定義收益窗口 w2:回看窗口
   3. WCutReturn.py(停用)
      - brief: 大单交易活跃（单笔成交金额高）的交易日，涨跌幅因子有更强的反转特性；对于大单交易不活跃（单笔成交金额低）的交易日，涨跌幅因子有更弱的反转特性。
      - subclass: [big_amount_ret, small_amount_ret]
      - param: r: w1 : 定义收益的窗口 w2 : 定义回看单笔交易 amount 的窗口
2. TickInflux
   1. Influx
      - brief: VPIN
      - subclass: [influx]  
      - param: ns: 移动平均窗口
   2. AverageInfluxAmount
      - brief: 平均单笔流出金额占比。股票下跌时，如果单笔成交金额大，说明委买有大单，是一种抄底行为。
      - subclass: [average_influx_amount, ma_average_influx_amount]  
      - param: ns:w1 : 定义上涨下跌窗口 return window 建议与预测目标收益区间同 w2 : 加总回看窗口 原为整日 w3 : 平滑窗口 （原为回看 n 天，与换仓频率相依）
3. TickMomentum
   1. HullMA
         - brief: https://kknews.cc/news/emejyjr.html
         - subclass: 
         - param: n: 移动平均窗口
   2. MidPriceEMADiff
      - brief: Difference Between Current Mid Price and EWMA and vwap of Mid Price of Window W
      - subclass: [原始 mid 的結果, wmid 的結果, vwap的结果]  
      - param: n: 移动平均窗口
   3. Momentum
      - brief: 各种历史收益, This Signal Calculates Log_Return_Signal, Return_Angle_Signal, Return_Acceleration_Signal
      - subclass: [Log_Return_Signal, Return_Angle_Signal, Return_Acceleration_Signal]  
      - param: ns: w1期收益, w2期窗口平滑, ret_scale：放大参数
   4. Reactivity
      - brief: This Signal Calculates Reactivity Signal By AI Gietzen(Market Reactivity - Automated Trade Signals)
      - subclass: 
      - param: ns: w1期窗口取值, w2期窗口平滑
   5. VolumeScaledMomentum
      - brief: This Signal Calculates The Scaled Return Times Scaled Volume Over Look Back Window w1
      - subclass: 
      - param: ns: w1期窗口取（收益）值, w2期窗口平滑
5. TickStrength
   1. MicroTorque  
      - brief: micro price 与 mid price 的差，理解为盘口价格的压力  
      Sasha Stoikov (2018) The micro-price: a high-frequency estimator of future prices, Quantitative Finance, 18:12, 1959-1966, DOI: 10.1080/14697688.2018.1489139  
      https://www.ma.imperial.ac.uk/~ajacquie/Gatheral60/Slides/Gatheral60%20-%20Stoikov.pdf
      - subclass: [MicroTorque] 
      - param: ns: 移动平均窗口
   2. SpreadFlip
      - brief: 盘口叠单
      - subclass: [af, bf, bf - af] 
      - param: n: 回看窗口
   3. BookImbalance
      - brief: 買賣量壓差    
        天风证券-天风证券市场微观结构探析系列之二：订单簿上的alpha
      - subclass: [五檔買賣壓, 十檔買賣壓] (觀察深度)
      - param: n: 回看窗口
   4. ConsumptionRate
      - brief: the Difference of Consumption Rate of Ask Orders over Consumption Rate of Bid Orders
      - subclass: 
      - param: n: 回看窗口
   5. SimpleOrderBookImbalance
      - brief: 買賣兩邊的 volume weighted price 與 midp 的差，以及兩者的差
      - subclass: ta.EMA(ask_dist, n), ta.EMA(bid_dist, n), ta.EMA(sig, n)
      - param: n: 回看窗口
   6. WeightedMidPriceDiff
      - brief: 量加權的 mid price 與 midp 的差，
      - subclass: [WeightedMidSignal, WeightedMidEMASignal, Modified_MidPriceRtn_Signal]
      - param: n: 回看窗口
6. TickVolatility
   1. CumMinMaxSignal
      - brief: 窗口最高最低价出现到现在的价差
      - subclass: [High_Price_Signal(价), Low_Price_Signal(价), 
                   High_Price_Time_Diff_Signal(时间), Low_Price_Time_Diff_Signal(时间)] 
      - param: n: 回看窗口

   2. SpreadTickRatio
      - brief: 盘口大小
      - subclass: [与 midp 的差, 与 wmid 的差]
      - param:
   3. VolatilityHL
      - brief: 过去一段时间的高低差与其衍生
      - subclass: [(1 - bh / p) * 10000,
                  (1 - bl / p) * 10000,
                  (dhl / p) * 10000,
                  (p / bhl - 1) * 10000,
                  rbh,
                  rbl] 
      - param: n: 回看窗口
   4. DIffVolatility
        - brief: 先差分再计算标准差構建每笔成交量差分标准差因子    
          高频因子（9）：高频波动中的时间序列信息_高频因子系列报告_长江证券)
        - subclass: [diff_std, diff_abs_mean]
        - param: n: w1:计算 std 窗口 w2:均线平滑窗口
   5. Skewness
      - brief: 高频偏度   
      《Does Realized Skewness and Kurtosis Predict the Cross-Section of Equity Returns?》Amaya et al. (2011)
      - subclass: [skew, ma_skew, downward_skew, ma_downward_skew]
      - param: n: w1:取计算 return 的窗口, w2:取样算偏度的窗口，可以常设 w1 与预测目标对齐，w2 采样不足没有代表性
   6. UniformityInformationDistribution
      - brief: 股价波动率大小的变化幅度，可以用来衡量信息冲击的剧烈程度。    
        “波动率选股因子”系列研究（二）：信息分布均匀度，基于高频波动率的选股因子_2020-09-01_东吴证券
      - subclass: [std, std（Vol_daily）除以 mean（Vol_daily）]
      - param: n: w1:定義收益的窗口 可以與預測目標保持一致 w2:定義波動率的窗口 原文為整日數據 因此此處應該遠大於 w1 窗口
   7. UniformityTurnoverDistribution
      - brief: std（TurnVol_daily）除以 mean（TurnVol_daily） 构建换手率分布均匀度    
        “技术分析拥抱选股因子”系列研究（四）：换手率分布均匀度，基于分钟成交量的选股因子_东吴证券
      - subclass: [std, std（TurnVol_daily）除以 mean（TurnVol_daily）]
      - param: n: w1:定義收益的窗口 可以與預測目標保持一致 w2:定義波動率的窗口 原文為整日數據 因此此處應該遠大於 w1 窗口
   8. VolumeStability
      - brief: 换手率稳定性    
        “技术分析系拥抱选股因子”系列研究（七）：量稳换手率选股因子：量小、量缩，都不如量稳？_2021-05-16_东吴证券   
        日内交易特征稳定性与股票收益——因子选股系列之四十九-东方证券
        《市场微观结构剖析系列之一：分钟线的尾部特征_2019-12-25_方正证券》
      - subclass: [turnover_rate, pct_turnover, std, std_std, hhi_std]
      - param: w1 : 第一個 STD 窗口 w2 : 第二個 STD 窗口
   9. ValueAtRisk
      - brief: VaR 的定义为：在一定的概率约束下和给定持有期间内，某金融投资组合的潜在最大损失值
      - subclass: [VaR, vwar_VaR]
      - param: w1 : 定义收益的窗口建议与预测区间相近 w2:定义分布采样的窗口 confidence : VaR 的信赖区间
   10. VolumeWeightedSkewness
       - brief: 成交量加权的方式计算过去一段时间价格分布的偏离程度，如果负偏态越明显，则个股在价格高位成交越多   
         高频因子（8）：高位成交因子——从量价匹配说起_高频因子系列报告_长江证券
       - subclass: [res]
       - param: w1 :回看 skewness 窗口
7. TickVolume
   1. MoneyFlowIndex
      - brief: 用買賣價量的變化描述買賣方的現金流
      - subclass: [(sma_u - sma_d) / (sma_u + sma_d)] 
      - param: n: 回看窗口
   2. QuoteChange    
      - brief: Difference Between Ask Cancel And Bid Cancel, Then Normalized By Its STDV
      - subclass: [signal_value] 
      - param: n: 回看窗口
   3. TotalVolumeChange    
      - brief: Total Volume(amount) Change From t-w1 To t, Scaled By Its EMA, 分速度與加速度 (一階差、二階差)
      - subclass: [Total_Volume_Change_Signal(一階差), 
                   Traded_Volume_Acceleration_Signal(二階差),
                   Total_amount_Change_Signal(一階差),
                   Traded_amount_Acceleration_Signal(二階差)]
      - param: ns: 回看窗口
   4. AmihudIlliquidityMeasure
      - brief: Amihud 非流动性因子     
        多因子alpha系列报告：高频价量数据的因子化方法_2021-07-12_广发证券
      - subclass: 兩家研報構建日內因子的不同方法
      - param: n: w1 收益率回看窗口 w2:平滑窗口
   5. AveragePriceBias.py
      - brief: 描述在相對高為成交的量價關係    
        因子选股系列(六十)--基于量价关系度量股票的买卖压力_东方证券    
        高频因子（8）：高位成交因子——从量价匹配说起_高频因子系列报告_长江证券
      - subclass: 兩家研報構建日內因子的不同方法
      - param: n: 回看窗口
   6. BuyInMovement
      - brief: 每日积极买入除以保守买入量   
        长江金工高频识途系列(一)_基于买入行为构建情绪因子_2017-03-15_长江证券
      - subclass: [BM := ∑CB/∑PB]
      - param: w1 :回看窗口
   7. DeltaVolumeCorr
      - brief: 每日先将该股票的分钟收盘价序列做一阶差分，再计算∆Pt序列与 Pt+1序列的相关系数   
      “技术分析拥抱选股因子”系列研究（五）：CPV因子移位版，价量自相关性中蕴藏的选股信息_东吴证券
      - subclass: [与下一期 t+1 回归, 与 t+w1 回归]
      - param: w1 : 差分窗口 w2 : 自回归窗口
   8. StructuredReverse
      - brief: 成交越活跃价格容易反转，给出按成交量加权的高频反转因子构建方式    
        高频因子（2）：结构化反转因子__高频因子系列报告_长江证券
      - subclass: 
      - param: w1 : ret 计算窗口 建议与预测目标相近 w2 : 平滑窗口
   9. VolumeCorr
       - brief: 量价相关性    
       “技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子_2020-02-23_东吴证券    
       选股因子系列研究（六十九）：高频因子的现实与幻想_2020-07-30_海通证券
       - subclass:[corr, corr_ma, corr_std]
       - param: w1: corr窗口 w2: ema 窗口
   10. VolumeEntropy
       - brief: 成交额熵      
       高频因子（8）：高位成交因子——从量价匹配说起_高频因子系列报告_长江证券 (成交额熵)
       - subclass:[amount 計算, volume 計算]
       - param: w1 : 回看窗口
8. TransInfo


##  4. TODO
1. Bar 的因子
   - “技术分析拥抱选股因子”系列研究（二）：上下影线，蜡烛好还是威廉好？.pdf
   - 天风证券-天风证券市场微观结构探析系列之三：分时K线中的alpha.pdf
































