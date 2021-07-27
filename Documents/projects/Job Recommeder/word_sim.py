# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:00:26 2021

@author: Pushkar
"""
#resume modeling:
import pandas as pd
keywords = {'ux,designer': ['design',
  'experi',
  'product',
  'user',
  'work',
  'ux',
  'team',
  'ui',
  'interact',
  'research',
  'custom',
  'everi',
  'happi',
  'revenue',
  'benefit',
  'modern',
  'mean',
  'reput',
  'financi',
  'statu',
  'account',
  'hold',
  'busi',
  'custom',
  'way',
  'platform'],
   'full,stack':['stack',
                 'ful','develope','web','fron','end',
                 'back','api','framework'],
   
 'data,scientist': ['data',
  'experi',
  'learn',
  'work',
  'team',
  'model',
  'product',
  'wish',
  'busi',
  'coursera',
  'learn',
  'product',
  'skill',
  'educ',
  'machin',
  'onlin',
  'applic',
  'recommend',
  'teach',
  'wish',
  'commerc',
  'transact',
  'environ',
  'market',
  'analyt',
  'mobil',
  'descript',
  'experi',
  'million'],
 'data,analyst': ['data',
  'experi',
  'work',
  'busi',
  'team',
  'analysi',
  'market',
  'report',
  'analyst',
  'analyt',
  'diagnost',
  'scientif',
  'experiment',
  'algorithm',
  'research',
  'polici',
  'assist',
  'job',
  'report',
  'studi',
  'coordin',
  'draft',
  'public',
  'analys'],
 'project,manager': ['project',
  'manag',
  'work',
  'requir',
  'job',
  'experi',
  'insur',
  'develop',
  'year',
  'capabl',
  'custom',
  'provid',
  'reconstruct',
  'accuraci',
  'person',
  'growth',
  'estimates',
  'assist',
  'bid',
  'strong',
  'capabl',
  'skill',
  'custom'],
 'product,manager': ['product',
  'experi',
  'work',
  'manag',
  'custom',
  'team',
  'busi',
  'develop',
  'data',
  'lend',
  'day',
  'success',
  'feedback',
  'successful',
  'project',
  'technic',
  'peopl',
  'world',
  'app',
  'startup',
  'founder',
  'anni',
  'engag',
  'fair',
  'solut',
  'build'],
 'big data engineer':['big','ware','house','data','wrangl','hadoo','spark',
                      'hdfs','mart','etl'],
 'data visualisation expert':['chart','dashboar','matplot','seabor','ui'],
 'artificial intelligence, deep learning': ['math', 'statistics', 'probability', 'predictions',
                             'calculus', 'algebra', 'Bayesian', 'algorithms', 'logic',
                              'physics', 'mechanics', 'cognitive', 'learning theory', 'language processing'],
 'data quality manager':['negoti','report','write','service','communica','analyt'],
 
 'account,manager': ['client',
  'manag',
  'custom',
  'insur',
  'sale',
  'account',
  'work',
  'servic',
  'time',
  'compani',
  'sale',
  'commiss',
  'custom',
  'floor',
  'mgr',
  'multifamili',
  'monthli',
  'priorit',
  'medidata',
  'sale',
  'devic',
  'strateg',
  'digit',
  'applaus',
  'revenu',
  'experi',
  'sell'],
 'consultant': ['consult',
  'client',
  'market',
  'work',
  'experi',
  'busi',
  'manag',
  'project',
  'team',
  'develop',
  'market',
  'healthcar',
  'data',
  'segment',
  'associ',
  'product',
  'model',
  'conduct',
  'industri',
  'test',
  'deloitt',
  'market',
  'autom',
  'solut',
  'architectur',
  'affili',
  'experi',
  'end',
  'tool'],
 'marketing': ['market',
  'manag',
  'experi',
  'work',
  'custom',
  'develop',
  'campaign',
  'content',
  'team',
  'brand',
  'growth',
  'optim',
  'impact',
  'team',
  'busi',
  'custom',
  'nurtur',
  'hire',
  'ori',
  'recruit',
  'facebook',
  'candid',
  'vici',
  'screen',
  'focus',
  'potenti',
  'pay'],
 'sales': ['sale',
  'custom',
  'insur',
  'work',
  'requir',
  'experi',
  'manag',
  'year',
  'team',
  'job',
  'insur',
  'brother',
  'plumb',
  'hvac',
  'paid',
  'diploma',
  'appli',
  'prefer',
  'order',
  'custom',
  'sale',
  'inventori',
  'maintain',
  'product',
  'price',
  'fulfil',
  'support']}

import process as pda

def resume_reader(example, keyword):
    toke_example = pda.tokenize_stem(example)
    example_words = toke_example[0].split()
    matching_words = []
    missing_words = []
    for word in keywords[keyword]:
        if word in example_words:
            matching_words.append(word)
        else:
            missing_words.append(word)
    return set(matching_words), set(missing_words)