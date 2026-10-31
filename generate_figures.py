from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import patheffects

ROOT=Path(__file__).parent; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
# Consistent publication palette
NAVY='#142B4A'; BLUE='#2878B5'; CYAN='#55B7C5'; TEAL='#168C83'; GREEN='#4D9B75'; AMBER='#D58A28'; RED='#C65353'; INK='#243447'; MUTED='#607286'; PALE='#F3F6FA'; LINE='#CAD4DF'; WHITE='#FFFFFF'; PURPLE='#7966A8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':12,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':320})

def rounded(ax,x,y,w,h,fc=WHITE,ec=LINE,lw=1.1,r=.025):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.012,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=lw)
 ax.add_patch(p); return p

def txt(ax,x,y,s,fs=9,c=INK,weight='normal',ha='center',va='center',**kw):
 if weight in ('left','right','center'):
  ha,weight=weight,'normal'
 ax.text(x,y,s,fontsize=fs,color=c,weight=weight,ha=ha,va=va,**kw)

def conn(ax,a,b,c=MUTED,lw=1.5,rad=0,style='-|>'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=13,linewidth=lw,color=c,connectionstyle=f'arc3,rad={rad}'))

def panel(ax,x,y,w,h,title,sub=None,fc=PALE,accent=BLUE):
 rounded(ax,x,y,w,h,fc=fc,ec=LINE)
 ax.add_patch(FancyBboxPatch((x,y),.009,h,boxstyle='round,pad=0,rounding_size=.004',facecolor=accent,edgecolor=accent,lw=0))
 txt(ax,x+.022,y+h*.68,title,9,INK,'bold','left')
 if sub: txt(ax,x+.022,y+h*.32,sub,7.2,MUTED,'left')

def save(fig,name):
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/f'{name}.png',bbox_inches='tight',dpi=320,facecolor='white')
 plt.close(fig)

# FIG 1: Graphical abstract / end-to-end architecture
fig,ax=plt.subplots(figsize=(16,6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.025,.95,'END-TO-END OFFLINE HANDWRITTEN TEXT RECOGNITION',17,NAVY,'bold','left')
txt(ax,.025,.895,'Document-specific conditioning  →  gated visual encoding  →  sequence modelling  →  CTC transcription',9,MUTED,'left')
# stage containers
for x,w,title,col in [(.025,.245,'01  DOCUMENT CONDITIONING',BLUE),(.29,.47,'02  RECOGNITION BACKBONE',TEAL),(.78,.195,'03  SEQUENCE DECODING',AMBER)]:
 rounded(ax,x,.20,w,.60,fc='#FBFCFE',ec=col,lw=1.35)
 txt(ax,x+.018,.755,title,8.3,col,'bold','left')
# left pipeline
panel(ax,.05,.57,.095,.12,'Input page','grayscale image',fc='#EAF1F8',accent=BLUE)
panel(ax,.16,.57,.095,.12,'Scale estimate','median component height',fc='#EAF1F8',accent=BLUE)
conn(ax,(.145,.63),(.16,.63),BLUE); conn(ax,(.255,.63),(.275,.63),BLUE)
panel(ax,.05,.37,.095,.12,'Normalize','illumination + skew',fc='#EAF1F8',accent=BLUE)
panel(ax,.16,.37,.095,.12,'Extract','lines / text regions',fc='#EAF1F8',accent=BLUE)
conn(ax,(.207,.57),(.097,.49),BLUE,rad=.15); conn(ax,(.145,.43),(.16,.43),BLUE)
txt(ax,.15,.275,'Scale-aware morphology and\nbranch-specific geometric rules',7.3,MUTED)
conn(ax,(.255,.43),(.305,.43),BLUE)
# middle backbone
x0=.315; gap=.012; widths=[.058,.058,.058,.058,.058,.058]
labels=[('Conv 1','8 ch'),('Conv 2','16 ch'),('Conv 3','32 ch'),('Conv 4','64 ch'),('Conv 5','80 ch'),('Conv 6–7','112 → 128 ch')]
for i,((t,s),w) in enumerate(zip(labels,widths)):
 x=x0+i*(w+gap); panel(ax,x,.55,w,.15,t,s,fc='#E9F5F2',accent=TEAL)
 if i<5: conn(ax,(x+w,.625),(x+w+gap,.625),TEAL,lw=1.2)

# lower contextual stages
panel(ax,.365,.30,.105,.13,'2-head MHSA','global context',fc='#EDF1FA',accent=PURPLE)
panel(ax,.50,.30,.115,.13,'3 × BiLSTM','128 units / direction',fc='#EDF1FA',accent=PURPLE)
panel(ax,.645,.30,.09,.13,'Dense','C logits',fc='#FFF4E5',accent=AMBER)
conn(ax,(.70,.55),(.415,.43),TEAL,rad=.12); conn(ax,(.47,.365),(.50,.365),PURPLE); conn(ax,(.615,.365),(.645,.365),PURPLE)
txt(ax,.515,.245,'Feature map → sequence',7.2,MUTED)
# decoder
panel(ax,.805,.57,.145,.12,'CTC loss','training objective',fc='#FFF4E5',accent=AMBER)
panel(ax,.805,.37,.145,.12,'Beam search','width = 32',fc='#FFF4E5',accent=AMBER)
conn(ax,(.735,.365),(.805,.63),AMBER,rad=-.10); conn(ax,(.735,.365),(.805,.43),AMBER,rad=.02)
rounded(ax,.82,.225,.115,.075,fc='#FFF9EF',ec=AMBER)
txt(ax,.877,.263,'TRANSCRIPT',8.2,NAVY,'bold'); txt(ax,.877,.238,'decoded text',6.8,MUTED)
conn(ax,(.877,.37),(.877,.30),AMBER)
txt(ax,.5,.09,'Six residual gated blocks are interleaved in the encoder  |  IAM Combined: C = 83, 1,583,847 parameters',8.5,NAVY,'bold')
save(fig,'fig01_graphical_abstract')

# FIG 2: conditioning flowchart with decision branches
fig,ax=plt.subplots(figsize=(14,7)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.025,.96,'SCALE-ADAPTIVE DOCUMENT CONDITIONING',16,NAVY,'bold','left')
txt(ax,.025,.915,'The estimated component scale controls selected morphology and geometric thresholds; extraction then follows the detected page structure.',8.8,MUTED,'left')
# Main sequence top
steps=[('Image input','grayscale + polarity'),('Estimate scale','median eligible CC height'),('Correct illumination','odd kernel: max(15, 4 × med_h)'),('Deskew and crop','detected skew + ink bounds'),('Suppress rules','horizontal opening / subtraction')]
xs=[.035,.225,.415,.605,.795]
for i,(a,b) in enumerate(steps):
 panel(ax,xs[i],.68,.155,.14,a,b,fc='#EAF1F8',accent=BLUE)
 if i<4: conn(ax,(xs[i]+.155,.75),(xs[i+1],.75),BLUE)
# branch split
conn(ax,(.872,.68),(.872,.56),TEAL)
rounded(ax,.39,.46,.22,.10,fc='#EAF5F2',ec=TEAL,lw=1.3); txt(ax,.50,.515,'Page structure / layout test',9,TEAL,'bold'); txt(ax,.50,.48,'single line or multi-region document?',7.1,MUTED)
conn(ax,(.872,.56),(.61,.51),TEAL,rad=.12)
# branches
panel(ax,.045,.23,.255,.16,'Single-line bypass','Small crop or wide line with a single-line projection profile',fc='#EEF3FA',accent=BLUE)
panel(ax,.372,.23,.255,.16,'Dense multi-line route','Projection valleys; at least six slices; branch-specific geometric filter',fc='#EAF5F2',accent=TEAL)
panel(ax,.699,.23,.255,.16,'Sparse card / note route','RLSA region grouping; branch-specific geometric filter',fc='#FFF4E5',accent=AMBER)
conn(ax,(.43,.46),(.18,.39),BLUE,rad=-.08); conn(ax,(.50,.46),(.50,.39),TEAL); conn(ax,(.57,.46),(.82,.39),AMBER,rad=.08)
for x in [.172,.499,.826]:
 conn(ax,(x,.23),(x,.145),MUTED)
rounded(ax,x-.105,.075,.21,.065,fc='#F4F6F8',ec=LINE); txt(ax,x,.108,'Resize / pad for HTR',7.8,NAVY,'bold')
txt(ax,.5,.025,'Important: geometric filters are deterministic heuristics, not a learned text/non-text classifier.',7.7,RED,'bold')
save(fig,'fig02_conditioning_decision_flow')

# FIG 3: architecture as layered signal path with explicit gate placement
fig,ax=plt.subplots(figsize=(15,6.3)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.025,.96,'RECOGNITION NETWORK: FROM IMAGE FEATURES TO CHARACTER SEQUENCES',15,NAVY,'bold','left')
txt(ax,.025,.915,'Architecture reconstructed from the implementation; channel widths and recurrent dimensions are shown explicitly.',8.8,MUTED,'left')
# feature extraction stack
rounded(ax,.025,.40,.49,.42,fc='#F8FBFD',ec=BLUE,lw=1.3); txt(ax,.045,.775,'VISUAL FEATURE EXTRACTION',8.5,BLUE,'bold','left')
chs=[8,16,32,64,80,112,128]
for i,c in enumerate(chs):
 x=.055+i*.064
 h=.12+(i/6)*.16; y=.51+(i%2)*.025
 rounded(ax,x,y,.052,h,fc='#E7F1F8',ec=BLUE,lw=1)
 txt(ax,x+.026,y+h*.62,f'{c}',9,NAVY,'bold'); txt(ax,x+.026,y+h*.30,f'C{i+1}',6.7,MUTED)
 if i<6: conn(ax,(x+.052,y+h*.5),(x+.064,y+.12+((i+1)%2)*.035),BLUE,lw=1)
 if i<6: txt(ax,x+.026,.465,'G',7,TEAL,'bold')
txt(ax,.27,.345,'7 convolution stages: 3×3 Conv + BatchNorm + Swish',7.3,MUTED)
txt(ax,.27,.305,'6 residual GatedConv2D blocks interleaved',7.3,TEAL,'bold')
# downstream
rounded(ax,.55,.34,.42,.48,fc='#FBFAFE',ec=PURPLE,lw=1.3); txt(ax,.57,.775,'SEQUENCE MODELLING AND OUTPUT',8.5,PURPLE,'bold','left')
panel(ax,.585,.57,.105,.13,'Reshape','feature map → T steps',fc='#EDF1FA',accent=PURPLE)
panel(ax,.72,.57,.105,.13,'MHSA','2 heads; key dim 8',fc='#EDF1FA',accent=PURPLE)
panel(ax,.855,.57,.09,.13,'BiLSTM','3 layers',fc='#EDF1FA',accent=PURPLE)
conn(ax,(.515,.58),(.585,.635),BLUE); conn(ax,(.69,.635),(.72,.635),PURPLE); conn(ax,(.825,.635),(.855,.635),PURPLE)
panel(ax,.64,.39,.22,.12,'Dense character projection','C dataset-specific logits',fc='#FFF4E5',accent=AMBER)
conn(ax,(.90,.57),(.79,.51),PURPLE,rad=.08)
rounded(ax,.64,.19,.22,.095,fc='#FFF9EF',ec=AMBER); txt(ax,.75,.245,'CTC LOSS / BEAM SEARCH',8.5,NAVY,'bold'); txt(ax,.75,.215,'training / inference',7,MUTED)
conn(ax,(.75,.39),(.75,.285),AMBER)
txt(ax,.5,.08,'For IAM Combined: C = 83  |  Dense projection = 256C + C parameters',9,NAVY,'bold')
save(fig,'fig04_recognition_architecture')

# FIG 4: residual gate math diagram
fig,ax=plt.subplots(figsize=(13,5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.025,.95,'RESIDUAL GATED CONVOLUTION: EXACT COMPUTATION',15,NAVY,'bold','left')
txt(ax,.025,.90,'The gate modulates the learned convolutional response before residual addition.',8.8,MUTED,'left')
# branches
panel(ax,.035,.56,.13,.16,'Input tensor','X',fc='#EAF1F8',accent=BLUE)
panel(ax,.235,.68,.16,.16,'3 × 3 convolution','S(X)',fc='#EAF1F8',accent=BLUE)
panel(ax,.455,.68,.16,.16,'Scalar modulation','γ S(X)',fc='#EDF1FA',accent=PURPLE)
panel(ax,.675,.68,.13,.16,'Sigmoid','σ(γ S(X))',fc='#EDF1FA',accent=PURPLE)
panel(ax,.455,.39,.19,.15,'Elementwise gate','G(X) = S(X) ⊙ σ(γS(X))',fc='#EAF5F2',accent=TEAL)
panel(ax,.70,.39,.12,.15,'Scale','β G(X)',fc='#FFF4E5',accent=AMBER)
panel(ax,.855,.56,.115,.16,'Residual sum','Y = X + βG(X)',fc='#EAF5F2',accent=TEAL)
conn(ax,(.165,.64),(.235,.76),BLUE); conn(ax,(.395,.76),(.455,.76),PURPLE); conn(ax,(.615,.76),(.675,.76),PURPLE)
conn(ax,(.74,.68),(.56,.54),TEAL,rad=.1); conn(ax,(.395,.76),(.455,.465),BLUE,rad=.08)
conn(ax,(.645,.465),(.70,.465),AMBER); conn(ax,(.82,.465),(.855,.60),AMBER)
conn(ax,(.165,.60),(.855,.60),MUTED,rad=-.22)
rounded(ax,.21,.12,.58,.13,fc='#F5F7FA',ec=LINE)
txt(ax,.5,.205,r'$G(X)=S(X)\odot\sigma(\gamma S(X))$',11,NAVY,'bold')
txt(ax,.5,.155,r'$Y=X+\beta G(X)$',11,NAVY,'bold')
txt(ax,.5,.055,'β and γ are learned scalar parameters in residual mode; this is not a pixel-level foreground mask.',8,RED,'bold')
save(fig,'fig03_residual_gate')

# FIG 5: data protocol as clean matrix
fig,ax=plt.subplots(figsize=(13,5.5)); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
txt(ax,.025,.95,'EXPERIMENTAL COVERAGE AND EVIDENCE STATUS',15,NAVY,'bold','left')
cols=[.03,.235,.365,.495,.625,.735,.84]; widths=[.20,.125,.125,.125,.10,.095,.13]
headers=['Dataset','Train','Validation','Test','Classes','Init.','Evidence']
for x,w,h in zip(cols,widths,headers):
 rounded(ax,x,.79,w,.085,fc=NAVY,ec=NAVY,r=.008); txt(ax,x+w/2,.832,h,7.8,WHITE,'bold')
rows=[('Parzival','2,237','912','1,328','56','Fresh','Test + logs'),('Saint Gall','468','235','707','49*','Fresh','Notebook test'),('ORAND-CAR-A','1,708','301','3,784','14','Fresh','Test + logs'),('CVL Digits','1,073','189','6,698','14','Fresh','Test + logs'),('IAM Combined','Not reported','966','Not reported','83','Warm-start','Validation only')]
for r,row in enumerate(rows):
 y=.675-r*.115
 for j,(x,w,val) in enumerate(zip(cols,widths,row)):
  fc=('#EEF4FA' if r%2==0 else '#F8FAFC')
  if row[0]=='IAM Combined' and j==3: fc='#FCEEEE'
  rounded(ax,x,y,w,.092,fc=fc,ec=LINE,r=.006)
  txt(ax,x+w/2,y+.046,val,7.5,(RED if row[0]=='IAM Combined' and j==3 else INK), 'bold' if j in [0,6] else 'normal')
txt(ax,.03,.105,'* Saint Gall class count is noted in the notebook context; its exported checkpoint and tokenizer are unavailable for independent verification.',7.5,RED,'left')
txt(ax,.03,.065,'IAM Combined is a warm-started run: best validation edit distance 5.66% at epoch 12; no independent held-out test result is present.',7.5,MUTED,'left')
save(fig,'fig06_dataset_evidence_matrix')

# FIG 6: training histories
runs={'Parzival':'model_parzival_epochs.csv','ORAND-CAR-A':'model_orand-car-a_epochs.csv','CVL Digits':'model_cvl-digits_epochs.csv','IAM Combined':'model_iam_combined_epochs.csv'}
fig,axs=plt.subplots(2,2,figsize=(12.5,8))
for ax,(name,file) in zip(axs.flat,runs.items()):
 df=pd.read_csv(ROOT.parent/'saved_models'/file); x=df['epoch']
 ax.plot(x,df['dist']*100,marker='o',ms=3,lw=1.8,label='Training')
 if 'val_dist' in df:
  ax.plot(x,df['val_dist']*100,marker='s',ms=3,lw=1.8,label='Validation')
  idx=df['val_dist'].idxmin(); xv=x.iloc[idx]; yv=df['val_dist'].iloc[idx]*100
  ax.scatter([xv],[yv],marker='*',s=100,zorder=4,edgecolor='black',linewidth=.4)
  ax.annotate(f'{yv:.2f}% @ epoch {int(xv)}',(xv,yv),xytext=(8,9),textcoords='offset points',fontsize=7.5,weight='bold')
 ax.set_title(name,loc='left',weight='bold',color=NAVY,pad=8); ax.set_xlabel('Epoch'); ax.set_ylabel('Normalized edit distance (%)'); ax.grid(True,alpha=.22); ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False,fontsize=7.5,loc='best')
fig.suptitle('Optimization histories available in the archived epoch logs',fontsize=14,weight='bold',color=NAVY,y=.99)
fig.text(.5,.015,'IAM Combined starts from an IAM checkpoint; the curves are not directly equivalent to fresh-initialization runs.',ha='center',fontsize=8,color=RED)
fig.tight_layout(rect=[0,.04,1,.96]); save(fig,'fig08_training_dynamics')

# FIG 7: test metrics with separated axes and uncertainty note
fig,axs=plt.subplots(1,2,figsize=(13,5))
datasets=['Parzival','Saint Gall','ORAND-CAR-A','CVL Digits']; cer=[2.48,8.38,3.33,50.78]
axs[0].bar(datasets,cer,color=[BLUE,BLUE,BLUE,RED],width=.62)
axs[0].set_title('Character error rate',loc='left',weight='bold',color=NAVY); axs[0].set_ylabel('Macro-averaged normalized edit distance (%)'); axs[0].set_ylim(0,60); axs[0].tick_params(axis='x',rotation=15); axs[0].grid(axis='y',alpha=.2); axs[0].spines[['top','right']].set_visible(False)
for i,v in enumerate(cer): axs[0].text(i,v+1.2,f'{v:.2f}%',ha='center',fontsize=8.5,weight='bold')
wd=['Parzival','Saint Gall']; wer=[8.53,41.62]
axs[1].bar(wd,wer,color=[TEAL,TEAL],width=.55); axs[1].set_title('Word error rate (available datasets only)',loc='left',weight='bold',color=NAVY); axs[1].set_ylabel('Macro-averaged normalized edit distance (%)'); axs[1].set_ylim(0,50); axs[1].grid(axis='y',alpha=.2); axs[1].spines[['top','right']].set_visible(False)
for i,v in enumerate(wer): axs[1].text(i,v+1,f'{v:.2f}%',ha='center',fontsize=8.5,weight='bold')
fig.suptitle('Held-out test results by dataset',fontsize=14,weight='bold',color=NAVY,y=1.02)
fig.text(.5,-.015,'No pooled score is reported. ORAND-CAR-A and CVL Digits WER values were not available in the archived evaluation output.',ha='center',fontsize=8,color=MUTED)
fig.tight_layout(); save(fig,'fig09_test_performance')

# FIG 8: discrepancy + parameter count, clean paired analysis
fig,axs=plt.subplots(1,2,figsize=(13,4.8))
vals=[6.9343,50.78]; bars=axs[0].bar(['Best validation\nedit distance','Held-out test\nCER'],vals,color=[TEAL,RED],width=.52)
axs[0].set_ylim(0,60); axs[0].set_ylabel('Normalized edit distance / CER (%)'); axs[0].set_title('CVL Digits: validation–test gap',loc='left',weight='bold',color=NAVY); axs[0].grid(axis='y',alpha=.2); axs[0].spines[['top','right']].set_visible(False)
for b,v in zip(bars,vals): axs[0].text(b.get_x()+b.get_width()/2,v+1.2,f'{v:.2f}%',ha='center',weight='bold')
axs[0].text(.5,.82,'43.85 percentage-point gap',transform=axs[0].transAxes,ha='center',fontsize=9,weight='bold',color=RED)
C=np.array([14,56,83]); total=1562516+257*C
axs[1].barh(['C = 14','C = 56','C = 83'],np.repeat(1562516,3)/1e6,color=BLUE,label='Shared backbone')
axs[1].barh(['C = 14','C = 56','C = 83'],257*C/1e6,left=np.repeat(1562516,3)/1e6,color=AMBER,label='Output projection')
axs[1].set_xlim(0,1.72); axs[1].set_xlabel('Parameters (millions)'); axs[1].set_title('Parameter scaling with vocabulary size',loc='left',weight='bold',color=NAVY); axs[1].legend(frameon=False,fontsize=7.5); axs[1].grid(axis='x',alpha=.2); axs[1].spines[['top','right']].set_visible(False)
for i,v in enumerate(total): axs[1].text(v/1e6+.015,i,f'{v:,}',va='center',fontsize=8,weight='bold')
fig.tight_layout(); save(fig,'fig10_diagnostic_analysis')

# FIG 9: literature context table as a figure, avoids a late-floating wide table
fig,ax=plt.subplots(figsize=(15,5.6)); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
txt(ax,.02,.96,'SELECTED IAM BENCHMARKS REPORTED IN THE LITERATURE',14,NAVY,'bold','left')
txt(ax,.02,.91,'Contextual comparison only: training data, pretraining, decoding, and protocol differ across studies.',8.5,MUTED,'left')
cols=[.02,.29,.39,.49,.59,.70]; widths=[.26,.09,.09,.09,.10,.28]
heads=['Method','Year','CER (%)','WER (%)','Params','Protocol distinction']
for x,w,h in zip(cols,widths,heads): rounded(ax,x,.78,w,.08,fc=NAVY,ec=NAVY,r=.006); txt(ax,x+w/2,.82,h,7.8,WHITE,'bold')
rows=[('Self-Attention + CTC + LM','2021','2.75','--','22M','Internal + public data; character n-gram LM'),('TrOCR-Small','2023','4.22','--','62M','Pretrained visual and text Transformers'),('TrOCR-Base','2023','3.42','--','334M','Pretrained visual and text Transformers'),('TrOCR-Large','2023','2.89','--','558M','Pretrained visual and text Transformers'),('HTR-VT','2025','4.70','14.90','53.5M','Vision Transformer encoder'),('LoGo-HTR','2026','2.01','10.27','6.4M','External self-supervised handwriting; full IAM labels'),('This study: IAM Combined','2026','--','--','1.584M','Validation 5.66%; no held-out IAM test')]
for r,row in enumerate(rows):
 y=.675-r*.083
 for j,(x,w,val) in enumerate(zip(cols,widths,row)):
  fc='#FCEEEE' if r==6 else ('#F1F5F9' if r%2==0 else '#FAFBFC')
  rounded(ax,x,y,w,.066,fc=fc,ec=LINE,r=.004)
  txt(ax,x+w/2,y+.033,val,7.1,RED if r==6 else INK,'bold' if j==0 else 'normal')
txt(ax,.02,.045,'Do not interpret this table as a leaderboard. The present IAM Combined run has validation evidence only, so its test CER/WER are intentionally blank.',7.7,RED,'bold','left')
save(fig,'fig07_literature_context')

# CTC illustrative figure
fig,ax=plt.subplots(figsize=(12,3.8)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,.025,.93,'CTC ALIGNMENT COLLAPSE',14,NAVY,'bold','left')
path=['c','c','blank','a','a','t','blank']; xs=np.linspace(.12,.88,len(path))
for x,ch in zip(xs,path):
 rounded(ax,x-.045,.60,.09,.16,fc='#EAF1F8' if ch!='blank' else '#F2F4F7',ec=BLUE if ch!='blank' else MUTED)
 txt(ax,x,.68,ch,10,NAVY,'bold'); txt(ax,x,.625,'frame',6.5,MUTED)
for a,b in zip(xs[:-1],xs[1:]): conn(ax,(a+.045,.68),(b-.045,.68),MUTED,1)
rounded(ax,.20,.27,.25,.13,fc='#EAF5F2',ec=TEAL); txt(ax,.325,.34,'c  blank  a  t  blank',9,TEAL,'bold')
rounded(ax,.60,.27,.20,.13,fc='#FFF4E5',ec=AMBER); txt(ax,.70,.34,'c a t',12,NAVY,'bold')
conn(ax,(.5,.55),(.325,.40),TEAL); conn(ax,(.45,.335),(.60,.335),AMBER)
txt(ax,.5,.13,'Merge adjacent repeats, then remove blanks. Illustrative path only, not a model prediction.',8,MUTED)
save(fig,'fig05_ctc_alignment')

# overview contact sheet
from PIL import Image,ImageDraw
files=[OUT/n for n in ['fig01_graphical_abstract.png','fig02_conditioning_decision_flow.png','fig03_residual_gate.png','fig04_recognition_architecture.png','fig05_ctc_alignment.png','fig06_dataset_evidence_matrix.png','fig07_literature_context.png','fig08_training_dynamics.png','fig09_test_performance.png','fig10_diagnostic_analysis.png']]
thumbs=[]
for f in files:
 im=Image.open(f).convert('RGB'); im.thumbnail((600,270))
 card=Image.new('RGB',(620,310),'white'); card.paste(im,((620-im.width)//2,10)); ImageDraw.Draw(card).text((12,286),f.stem,fill='black'); thumbs.append(card)
cols=2; rows=(len(thumbs)+1)//2; sheet=Image.new('RGB',(cols*620,rows*310),(225,230,236))
for i,im in enumerate(thumbs): sheet.paste(im,((i%cols)*620,(i//cols)*310))
sheet.save(OUT/'FIGURES_OVERVIEW.png')
with open(OUT/'FIGURE_MANIFEST.md','w') as f:
 f.write('# Final figure set\n\nAll diagrams are generated as vector PDF and high-resolution PNG from the same source script. The PDF manuscript embeds the vector PDF files directly.\n\n')
 for n in ['fig01_graphical_abstract','fig02_conditioning_decision_flow','fig03_residual_gate','fig04_recognition_architecture','fig05_ctc_alignment','fig06_dataset_evidence_matrix','fig07_literature_context','fig08_training_dynamics','fig09_test_performance','fig10_diagnostic_analysis']:
  f.write(f'- `{n}.pdf` / `{n}.png`\n')
print('Generated',len(list(OUT.glob('fig*.pdf'))),'vector figures')
