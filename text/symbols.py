""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
#_letters_mandarin = '打远一看，它们的确很是美丽英国哲学家曾经说过“我老在北京哇塞！太精彩了。不管怎么主队好象志夺魁乘船漂游三峡真刺激喜欢迎你”每个月次电话沙尘暴像给人都带来麻烦就这意思又聪明收藏他肯定有妇女节快乐永爱妈谈恋现样周末只忙着陪谁也认识帮助特别网球和登山眼被迷住上大用得到见咱去那儿玩吧嘛为什要偷笑前几天碰件趣事已播至该专辑最后声音朋友刚从巴厘岛回做些吃促进宝食欲身影总心里晃觉并非品质狡猾常陷害工厂把直接排气中没错车门另外两问可啊除飞碟但能听清请时间提醒遇位亮孩青蛙冬眠马勇敢无辜会杀必须通才作妻子盼望您面当然饿死找日程走候锁制药公司短期姐空暖窝逗以选择年旅馆比方华尔街报知道生帆风顺轮流开台湾地区动物园早指示递纸花叫红包关系许婆泄露秘密应怕始段新感情基于智统创造实际形成氧化碳参加全变集跳舞难呀还正练本功让宽广魄凡坐铁需约二十分钟下结婚路跟告努力戒酒退晚拍点差先握钓鱼、聊野营深印脑海买玫瑰喝多伤体如行擅长卖萌令鼓资料她香甜微今第七教师想熊猫块头守员奏礼曲设计小巧单手操奥林匹克之父装饰镇所扯闲港叔吸引书腿健康放泳竟此处挺扭洋冰淇淋桃糟糕躺床果诉凑六再鸡肉神奇搞懂倒―古董发型蛮斗胆啦宿舍附近自昨起忧郁温蒸俩合吵架保持沉默表四哪种技较棒其己晨极玛挖苦口渴习惯或消遣尼迪寓等同捉弄星五考虑离市汤姆却赌博瘾魂颠返钱谢纹沃斯盾房芝哥待颜色适对而希吝啬鬼骑牵挂依东西更相乎运楚控配值嘴鹦鹉乳臭未干记者猜减肥痛响仍紧张双鞋喊布赖恩枪投降脚坑语普算松承蒙邀病讲理性格冷静且客观牺牲换取金班停原弗兰草莓妙元答传挥幸少服务器超套瓷珍贵竞活数百万刻失业惹丑闻折腾久转维申调雨概足够支绝准备零号凉输掉软妹：讨厌办惊派爬名光午泪滚出顿炖剩余八秒将高交各千秋净整洁阳朗堆照片衣购狂患哒呆咖啡洒桌歌养倦裁判满简拜平衡透部诱饵坚越汉森伦敦热随爸信受城噪厅场置扰棋艺托笔污染篮嗅鼻雪沿溪味价左右勿入竹诗穿究亲属贾突兴奋领男店脖类戏吴亦帅忽悠机初佣度漫黑夜句义白赛甩反厉慷慨淑文首款级删闹嘈杂沈雷阵云晴梅往视违法留注终庭院湖央安划乒乓充重联断复梦板滑录南舒宜振屏幕显银灰济著祸根祝春福撞筋苏械农内查怀孕半局休息杯拒束向借插源尚蒂娜洗详细绕晕效劳餐尽改完醋偶夏季仅连票兵式村夸奖鲍伯纪睛闪轻担责任乱龄互脸画缺拥挤木棍梯唐朝铸尖牙酷副脏—演唱'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
