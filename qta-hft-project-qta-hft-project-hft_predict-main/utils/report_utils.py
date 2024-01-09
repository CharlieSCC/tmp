#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListItem, ListFlowable, Frame
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
import pandas as pd
from utils.visualize_utils import main
from feature_analysis import Value2Rank
from glob import glob

def gen_report(factor, type='T0'):
    '''
    输入factor例子: T0momentum or T0Momentum.a or Momentum or Momentum.a
    '''
    factor = type + factor if type not in factor else factor
    mapping_score = {}
    for ii in range(l:=len(pd.read_csv('../output/t_values.csv', index_col=0).index)):
        mapping_score[ii] = 100*(1-ii/l)
    target = ['TickReturn', 'TickReturnLabelThres']
    rt = ReportTemplate()
    story = []
    title1 = f'{factor}因子报告'
    story.append(Paragraph(title1,style=rt.txt_style['标题1']))
    story.append(Spacer(240, 30))

    num = 7

    def DataFrame2Table(df):
        return [df.columns[:, ].values.astype(str).tolist()] + df.values.tolist()

    def Csv2Table(path, factor, num):
        df = pd.read_csv(path, index_col=0).sort_index()

        if '.' in factor:
            factor = factor.split('.')[0]
            ex = factor.split('.')[-1]
            factor_list = [x for x in df.index if factor in x and ex in x]
        else:
            factor_list = [x for x in df.index if factor in x]
        if len(rank_col:=[x for x in df.columns if 'rank' in x or 'Rank' in x])>0:
            table = df.loc[factor_list, rank_col].round(3)
        else:
            table = df.loc[factor_list].round(3)
        if (j:=table.shape[1]/num)<=1:
            table.reset_index(inplace=True)
            story.append(rt.gen_table(DataFrame2Table(table)))
            story.append(Spacer(240, 20))
        else:
            for _ in range(int(j)):
                temp_table = table.loc[:,table.columns[_*num:_*num+num]].reset_index()
                story.append(rt.gen_table(DataFrame2Table(temp_table)))
                story.append(Spacer(240, 20))
            temp_table = table.loc[:, table.columns[_ * num + num:]].reset_index()
            story.append(rt.gen_table(DataFrame2Table(temp_table)))
            story.append(Spacer(240, 20))
        return table

    def group_factor_impute(tb,refer_index):
        if tb.shape[0]==1:
            tb = pd.DataFrame(index=refer_index, columns=tb.columns, data=np.vstack([tb.values for _ in range(len(refer_index))]))
        else:
            temp_df = pd.DataFrame(index=refer_index, columns=tb.columns)
            name = [(x.split('.')[0],x.split('.')[-1]) for x in tb.index]
            for ind in temp_df.index:
                for ii in range(len(name)):
                    if name[ii][0] in ind and name[ii][1] in ind:
                        temp_df.loc[ind]=tb.iloc[ii]
                        break
            tb = temp_df
        return tb
    
    title2 = '1. 描述性统计'
    story.append(Paragraph(title2, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '数据统计性描述，分别计算了特征分布的均值、标准差、分位数、偏度、峰度以描述特征的分布情况: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    Csv2Table('../output/stat_description.csv', factor, num)

    if factor != 'T0':
        txt = '通过图形可视化描述数据的数值分布, 与预测目标收益率TickReturn的相关性， 以及特征在时序上的表现: '
        story.append(Paragraph(txt,style=rt.txt_style['正文']))
        if len(png_lt:= [x for x in glob('../output/*.png') if factor in x])==0:
            main(factor)
        # 改进：不同y对应的图
        story.append(rt.gen_img(png_lt[0]))

    title2 = '2. 相关系数'
    story.append(Paragraph(title2, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '分别计算给定不同参数和不同计算方式的特征与多个预测目标的相关系数，直接观察其线性关系: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    Csv2Table('../output/correlation.csv',factor,num)

    title3 = '3. t检验'
    story.append(Paragraph(title3, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '纳入全部的特征，分别对不同预测目标进行线性回归，观察其系数的显著性表现，可以一定程度上判断线性关系，同时也可以' \
          '帮助判断同一大类特征内不同参数的取舍: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    t_tb = Csv2Table('../output/t_values.csv',factor,num).set_index('index')

    title4 = '4. F检验'
    story.append(Paragraph(title4, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '纳入全部的特征，分别对不同预测目标进行线性回归，对某一大类的特征进行F检验，从而判断该类特征是否有线性预测能力: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    f_tb = Csv2Table('../output/f_values.csv', factor, num).set_index('index')
    ## 由于是group feature name所以要填充操作
    f_tb = group_factor_impute(f_tb, t_tb.index)

    title5 = '5. 熵-信息增益'
    story.append(Paragraph(title5, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '分别将连续的特征进行离散化，分为一百个标签，判断给定某一特征后，能给模型预测目标的熵带来的信息增益情况，' \
          '由于TickReturn为连续型变量，因而通过直接按正负性划分为两类判断: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    infor_tb = Csv2Table('../output/information_gain.csv', factor, num).set_index('index')

    title6 = '6. 分组指标评价'
    story.append(Paragraph(title6, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '将某大类因子放入模型，分别根据评价指标（R_square, F1_score)获取该类因子排名: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    group_tb = Csv2Table('../output/group_eval.csv', factor, num).set_index('index')
    group_tb = group_factor_impute(group_tb, t_tb.index)


    title7 = '7. 模型SHAP'
    story.append(Paragraph(title7, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '通过SHAP衡量各个特征的表现: '
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))
    shap_tb = Csv2Table('../output/shap_values.csv', factor, num).set_index('index')
 
    
    

    title10 = '10. 评分'
    story.append(Paragraph(title10, style=rt.txt_style['标题2']))
    story.append(Spacer(240, 20))
    txt = '特征的评分分为以下两部分: 线性部分和非线性部分。线性部分考纳线性回归的统计性结果，对相应的排名进行映射为分数，' \
          '最后将所有线性评分进行平均处理得到线性分数。非线性部分则参考了熵，SHAP，LGBM重要性等指标排名，同样进行映射平均。'
    story.append(Paragraph(txt,style=rt.txt_style['正文']))
    story.append(Spacer(240, 20))

    title10_1 = '10.1 线性评分'
    story.append(Paragraph(title10_1, style=rt.txt_style['标题3']))
    story.append(Spacer(240, 20))

    linear_score = pd.concat([t_tb, f_tb],axis=1).applymap(lambda x:mapping_score[x])
    score1=pd.DataFrame()
    for x in target:
        score1[x]=linear_score[[y for y in linear_score.columns if x == y.split('_')[0]]].mean(axis=1)
    score1.reset_index(inplace=True)
    story.append(rt.gen_table(DataFrame2Table(score1)))
    story.append(Spacer(240, 20))

    title10_2 = '10.2 非线性评分'
    story.append(Paragraph(title10_2, style=rt.txt_style['标题3']))
    story.append(Spacer(240, 20))

    nonlinear_score = pd.concat([infor_tb, group_tb, shap_tb],axis=1).applymap(lambda x:mapping_score[x])
    score2 = pd.DataFrame()
    for x in target:
        score2[x]=nonlinear_score[[y for y in nonlinear_score.columns if x == y.split('_')[0]]].mean(axis=1)
    score2.reset_index(inplace=True)
    story.append(rt.gen_table(DataFrame2Table(score2)))
    story.append(Spacer(240, 20))

    title10_3 = '10.3 综合评分'
    story.append(Paragraph(title10_3, style=rt.txt_style['标题3']))
    story.append(Spacer(240, 20))
    final_score = pd.merge(score1.set_index('index'),score2.set_index('index'),left_index=True, right_index=True)
    for x in target:
        final_score[x] = final_score[[y for y in final_score.columns if x==y.split('_')[0]]].mean(axis=1)
    final_score = final_score[target].round(3).reset_index()
    story.append(rt.gen_table(DataFrame2Table(final_score)))
    story.append(Spacer(240, 20))

    if factor == 'T0':
        Value2Rank(final_score.set_index('index')).to_csv('../output/factor_score.csv')
    else:
        doc = SimpleDocTemplate(f'../output/{factor}因子报告.pdf')
        doc.build(story)

class ReportTemplate:
    def __init__(self):
        self.txt_template()
        self.table_template()


    def txt_template(self):
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        # 下载字体，解决中文乱码
        pdfmetrics.registerFont(TTFont('SimKai','SimKai.ttf'))
        pdfmetrics.registerFont(TTFont('SimSun','SimSun.ttc'))
        self.txt_style = getSampleStyleSheet()
        self.txt_style.add(ParagraphStyle(name='正文', alignment=TA_JUSTIFY, fontName='SimKai', fontSize=12,
                                          textColor=colors.black,firstLineIndent=20,leading=24,spacebefore=1))
        self.txt_style.add(ParagraphStyle(name='标题1', alignment=TA_CENTER, fontName='SimKai', fontSize=20,
                                          textColor=colors.black, wordWrap='CJK'))
        self.txt_style.add(ParagraphStyle(name='标题2', alignment=TA_LEFT, fontName='SimKai', fontSize=14,
                                          textColor=colors.black, wordWrap='CJK'))
        self.txt_style.add(ParagraphStyle(name='标题3', alignment=TA_LEFT, fontName='SimKai', fontSize=13,
                                          textColor=colors.black, wordWrap='CJK'))

    def table_template(self):
        self.table_style = [
            ('FONTNAME', (0, 0), (-1, -1), 'SimKai'),  # 字体
            ('FONTSIZE', (0, 0), (-1, -1), 12),  # 第一行的字体大小
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),  # 设置表格内文字颜色
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),  # 设置表格框线
            ('LINEABOVE', (0, 1), (-1, 1), 1, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ]

        # self.table_style2 = [
        #     ('FONTNAME', (0, 0), (-1, -1), 'SimKai'),  # 字体
        #     ('FONTSIZE', (0, 0), (-1, -1), 12),  # 第一行的字体大小
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
        #     ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),  # 设置表格内文字颜色
        #     ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),  # 设置表格框线
        #     ('LINEABOVE', (0, 2), (-1, 2), 1, colors.black),
        #     ('LINEABOVE', (0, 4), (-1, 4), 1, colors.black),
        #     ('LINEABOVE', (0, 6), (-1, 6), 1, colors.black),
        #     ('LINEABOVE', (0, 8), (-1, 8), 1, colors.black),
        #     ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        # ]

    def gen_img(self, filepath):
        img = Image(filepath)
        img.drawHeight = 200
        img.drawWidth = 220
        return img

    def gen_table(self, table_data):
        table = Table(table_data,style=self.table_style)
        return table


if __name__ == "__main__":
    gen_report('')
    # gen_report('Momentum')
