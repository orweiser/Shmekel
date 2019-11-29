from ..core.dataset import Dataset
from Utils.data import DataReader
from shmekel_core.calc_models import Stock
from feature_space import get_feature
from copy import deepcopy as copy
import os
from tqdm import tqdm
from Utils.logger import logger


N_train = 300
N_val = 80


DEFAULT_TRAIN_STOCK_LIST = tuple(['anh','artx','brt','sfl','nsit','sbac','ssb','myj','cev','itrn','nly','gef','oiim','olp','fcbc','syt','igld','ltbr','pbr','ohi','bpop','rwt','sam','amn','cnxn','bwinb','fund','scs','hpf','mrvl','hsii','cac','clar','artna','ftf','wba','pcn','pkd','hrl','sjt','tenx','crf','vmc','pmo','lkfn','teso','gty','wpc','scg','avt','ctg',
                                  'mrtn','safm','ltrx','bt','lci','hbhc','hvt','bgfv','tear','ppbi','tef','sgms','fele','cbz','mua','clbs','hsic','rga','arcb','bgr','csu','lbtya','dnb','mlr','colm','htlf','nwl','fcel','ry','agco','myi','cern','cuba','vgz','star','bsrr','rtn','bsqr','www','hsc','cir','wilc','fizz','bcrx','bbf','mgm','ohai','nbh','mpwr','ghc',
                                  'cthr','jfr','cf','tpx','pgp','dwsn','px','obe','aig','cot','reis','axp','bco','crl','pmbc','vlt','aan','rvlt','jrs','dwch','sanm','lnc','cost','cna','pdex','su','holi','amgn','oflx','gut','nxtm','twx','psb','fll','elp','rmt','alco','aeg','intu','mfin','df','ges','ceqp','pnw','rpm','k','iosp','unfi','umpq','fis',
                                  'sif','mpaa','fco','ric','ci','ingr','xl','cinf','shlo','cia','hfwa','ntn','emf','mcn','epr','ns','pesi','atvi','ttmi','rvp','rdwr','sigi','dht','bpl','sri','ticc','swx','gigm','erh','amkr','cntf','mu','hps','do','myd','stc','gpk','apri','clgx','isig','fdef','inod','axas','mlm','adbe','an','trq','stra','nuv','trib',
                                  'idra','jout','fico','wst','ipas','mth','ubsh','mbtf','shen','uaa','cii','mmm','pep','spxx','flxs','frt','rfil','puk','cmo','trv','plpc','atlo','kim','csv','hygs','cycc','blw','cswc','dal','vsh','ts','pbh','fitb','pky','lll','ghdx','bios','dxpe','asna','mro','pcg','caw','tdy','cnq','kvhi','dfbg','hgt','oksb','yorw','wstl',
                                  'luk','ggg','elon','asml','a','ae','lstr','cffn','xper','cs','irix','rcii','phg','tpc','denn','rf','cy','eep','dltr','cidm','nrg','lyts','asr','mnp','clb','scl','hig','prk','mo','ty','btx','wts','agn','ccmp','snbr','gsh','icui','cbl','mdco','avid','mbfi','aoi','trn','ipar','msp','hain','glad','emms','wpcs','bmtc',
                                  'psti','ford','iiji','bcs','t','ntap','expo','eip','fr','cva','vlgea','lor','eqix','ehi','gps','azz','ypf','sivb','bac','ntgr','spa','rbcaa','bdn','rprx','gbnk','vicr','mmc','infi','mtd','tgs','viv','expe','rcky','kr','ntx','hrc','sina','ee','alk','mfl','dla','nus','kye','trmb','ccrn','cvgw','css','hska','belfb','caas',
                                  'wafd','cyrn','agm','jpm','tgi','cprt','ghm','plxs','pck','iaf','bidu','acgl','nhc','abr','bqh','enx','ag','wlk','frbk','bbt','sbcf','hmsy','fsi','lhcg','cyh','buse','cms','ocn','cree','chi','mmsi','qsii','dar','num','powi','vvi','ppc','efsc','vmo','smp','pvh','prsc','aem','pcf','akr','stkl','rqi','nad','hele','qdel',
                                  'hurc','nfj','cmpr','hr','bb','dcom','tecd','core','navg','mqy','fbp','fen','mdrx','rlh','cef','gbci','cyan','usb','bh','vpv','lsi','lpsn','txn','nwe','pkx','umc','alex','sne','acls','chs','arry','fwrd','bkt','kex','srce','mzf','marps','sgc','eiv','tlgt','nat','laws','endp','rrd','nea','iret','bbox','fe','foe','azn',
                                  'tot','chdn','pfl','sr','cacc','iart','cort','e','gil','np','sf','wgl','bke','hsbc','mei','rhp','pot','peo','tli','crvl','isca','bkmu','rgen','brc','bel','emd','rjf','pkg','aiz','heb','mhn','cxe','els','inve','tup','ncs','pds','ctrn','unp','bgc','sgen','knd','mmv','cub','ngd','krg','wex','bofi','mye','spxc',
                                  'jro','msm','d','kmg','wsbf','nbl','acxm','hibb','arlz','evv','cohr','wr','lpth','hni','wlfc','isl','ecpg','vrx','artw','fmc','web','auo','aphb','bcor','bfo','f','mlhr','gogl','pcom','crs','axgn','casc','sybt','sah','lub','dnp','rfi','alb','ubsi','culp','hum','twi','cenx','msex','alny','gpx','sxi','rol','nke','lpt',
                                  'nbix','flic','apf','ch','cma','avy','teva','sti','twn','lby','nvln','blj','cdns','pgc','gtxi','cohu','gss','cxw','aht','msa','pg','banf','mpa','ap','fac','ce','sptm','mpv','blk','wso','ecl','enz','ckh','uqm','dco','pki','hon','amrb','bgg','nxc','tgna','bro','eto','andv','esp','geos','mav','cohn','usa','abcb',
                                  'ibn','weys','hiw','san','natr','sig','tga','omex','fix','moh','eix','adi','twmc','emn','becn','tsn','msf','lad','jobs','jeq','tv','apc','bhp','hoft','arcc','dci','deo','sfe','cpf','smrt','cri','mgpi','pery','imkta','wrld','sna','ntc','sui','rave','skm','ktec','gain','wwr','fmx','dish','chn','valu','airt','stm','gpm',
                                  'amsf','recn','cbg','snak','alks','nro','trex','pcti','mind','hcsg','evf','mcd','gab','hcp','slb','wlb','rmti','pcm','eqc','mbt','puk_','iif','rbc','crmt','gabc','glp','tgb','xlnx','unt','bxs','nr','slp','ccur','iboc','shi','morn','plus','gme','tisa','bif','ncty','exfo','pcyg','gsbc','biol','tech','cbsh','iiin','cool','pfd',
                                  'kerx','foxa','tisi','syx','au','tzoo','hrb','rig','ktcc','svu','psmt','prcp','uonek','ceco','xoma','stmp','mdp','abio','cb','cif','ksm','amag','lcut','lscc','shoo','bte','mtrx','crr','pxd','ostk','ofc','bpt','gpor','asx','nuo','novt','cmd','giga','bebe','indb','bsac','r','nvax','teo','sup','msd','kyo','ivz','ppt','anss',
                                  'eeq','bwld','leo','mxc','wiw','wppgy','pho','oi','etg','amnb','epc','bc','apd','schl','zeus','bmy','bks','wtw','ufi','mark','bvx','ha','ngg','cpl','cls','vfl','mho','crt','ddf','efx','brn','ccu','teck','flr','jnpr','lng','mitk','sppi','rvt','ulbi','aste','regn','pbi','abc','fmo','arql','lyg','ucfc','nvo','epm',
                                  'silc','glv','ggn','alxn','calm','tsm','avb','fnf','nmrx','lho','ome','msn','myn','etv','slab','esl','intt','nok','ufpi','lmnx','nvda','fii','gch','nny','plg','gbli','cresy','bxmt','eei','udr','mksi','nbo','rushb','oln','crnt','shld','tgc','dlr','tnc','fcx','cp','bfk','txt','sbgi','kona','dte','mnkd','avd','spar','dvcr',
                                  'evc','wtba','oia','intl','smg','xpo','cvu','ugi','merc','chke','cpss','phk','vmi','mags','icb','ntwk','wnc','caci','evg','pei','lzb','fisi','ptnr','mtex','evo','cybe','aa','pii','inuv','vno','len','l','igd','csfl','alx','drq','rrc','forr','tds','csbk','slgn','ble','ddr','bxp','sa','bk','micr','nan','ndaq','odp',
                                  'atrs','myc','edap','intc','omn','xom','etb','schn','aaba','cce','dcm','bhe','ayi','strl','obci','giii','dre','epay','nyt','apog','fsbk','lly','nwy','bdj','issc','xec','cvo','rs','qumu','dov','nac','erf','vly','fbr','mtz','cyd','rht','znh','isr','bme','nrp','jmba','nmy','saft','etn','aiv','adsk','exk','ahl_c','ptie',
                                  'ths','tk','ko','wab','gfi','ebf','man','lvlt','sci','tss','bfz','ftr','mov','rop','leu','psec','dhr','tdf','mgln','arrs','eght','jblu','ptr','cah','cbs','nov','atlc','mnro','vbf','bp','ktos','nrk','srpt','smfg','akam','hnp','cuk','arw','akp','all','nnc','dorm','func','nmt','pnr','gv','pfbi','hqh','sga','fnlc',
                                  'bjri','ttek','ing','camt','wltw','mca','pcyo','mpvd','amp','rdn','kool','if','o','neog','unf','adc','etfc','etm','kf','wtt','dvax','mtsc','cvbf','staa','pir','cmi','fast','byd','cco','glt','ul','ocfc','cwt','ori','rci','mnr','hbm','csl','flc','pfs','fonr','kem','uvsp','sofo','aaxn','kbh','ppl','baa','genc','tex',
                                  'abe','amot','ntp','int','el','stly','axs','diod','hd','awf','bns','gilt','pke','egbn','snp','cgg','acc','cmcsa','xrm','mat','rt','dsu','eos','otiv','flws','casy','ozrk','uri','drys','mchx','cnbka','ebix','ajrd','atri','lrad','nxr','mon','eqs','abmd','wia','bxmx','lcm','cris','evol','oxy','snh','laq','inap','ftk','hlit',
                                  'de','pmx','fult','rog','useg','myl','sji','bpk','wwe','ncz','ufcs','nl','lvs','plab','spok','hyt','pmf','egn','tyl','seic','isrg','tif','mft','rhi','zbra','io','glw','nmi','grvy','sm','wec','zion','jwn','esca','cytx','on','cde','dx','dest','gra','pwr','abb','nci','usm','kfs','ibm','aos','cbb','ravn','jcp',
                                  'arkr','kopn','cvm','mtb','cdor','lndc','hurn','nath','wac','mfv','bbk','shg','nm','oled','amov','clm','gd','ccl','ois','wgo','wal','ncv','seb','xel','ggp','ctl','pcbk','vrtx','hall','ccne','nati','low','tcbi','cgen','bdc','pcar','peg','b','bfin','cmct','depo','shpg','nsl','dhil','utmd','gam','lway','halo','penn','mpx',
                                  'meip','wg','mbwm','osk','egy','dhy','wre','ful','bxc','dhi','bto','wbk','msft','lnce','afam','bont','aaon','un','ava','wcc','nqp','hlf','lbix','dni','tu','bnj','ilmn','pty','amat','cha','wmb','bjz','sp','fve','gold','nrim','htm','asa','auy','cts','eme','apdn','cwst','hp','che','fmn','wdc','dvn','cgnt','hal',
                                  'tmk','ssl','mt','fox','res','apu','uhs','stz','bsx','pcq','vsat','so','hban','opy','ida','ipg','acy','ofg','cutr','pru','ale','mpw','wrb','tlk','naz','pebo','bdx','frd','mrk','kgc','wsfs','emci','nvmi','itg','stba','vdsi','atni','oclr','mms','hos','mxim','rdcm','rsys','chco','nue','etw','fbc','immr','eea','jhg',
                                  'mfa','pol','gg','hep','hbio','linc','pfo','trxc','cog','msb','mnta','insg','rost','adm','utx','mcr','hrs','tbi','pets','cee','drad','crm','kt','itcb','tsu','brfs','veru','mtl_','cass','alsk','fbnc','irs','mic','ufpt','tile','syna','hpt','avp','usau','idt','zumz','fss','enb','naii','chkp','rnst','lnt','mqt','qtm','lcii',
                                  'kwr','ge','gec','slm','grow','adp','jcom','payx','tayd','rmbs','ed','clh','mus','vtn','feim','fra','imgn','dva','cl','umh','rwc','fti','sbi','sid','gbx','lake','ltm','erc','grr','cal','mtu','cm','nni','svra','fstr','gs','pope','ne','pten','ggal','rgs','bdsi','midd','ael','amg','sjw','kro','ead','mcbc','kst',
                                  'wy','iqi','swn','fls','hdsn','eoi','ai','ptn','wdr','sasr','husa','seac','bhlb','pzg','big','hql','m','wpz','amtd','yuma','pzc','labl','ipci','glng','hson','axr','ntl','cag','amsc','nls','lion','nbr','ggt','wm','synt','lamr','vvc','pso','nak','prgo','swks','aciw','efr','rad','bkn','nmz','hwkn','mstr','mag','vmm',
                                  'chk','abt','mmp','cbu','wneb','lpl','swk','mkl','oii','yum','cmls','dri','cvg','kyn','mos','iga','ame','exp','umbf','skt','brkl','meoh','nssc','bas','cea','frme','tei','hsy','gtn','geo','jakk','cmt','cmc','sjr','brkr','chy','cuz','lkq','elgx','chl','pra','idxg','dks','csq','lxrx','nbw','nuva','mvis','rl','x',
                                  'emj','lm','hpq','bset','oxm','ewbc','nycb','crai','mant','fsp','atr','pkoh','cien','odc','good','bkh','wsm','phiik','s','fdp','crk','drrx','gsb','shw','bbw','uhal','afg','cpsi','rio','dsgx','rmcf','cnsl','fosl','brk-b','vsec','jjsf','stld','ppr','chci','maa','cnmd','usg','aee','omc','pcln','njr','sph','emkr','crme','utg',
                                  'duk','asur','pfe','mvt','idxx','win','rev','arna','eltk','bam','anf','farm','dhf','vki','sny','dds','noc','pfg','pool','ctb','uwn','tlf','mkc','quik','nvs','ora','neu','cr','wwd','alog','cet','nkg','bvsn','bms','cpst','nim','min','mhi','rigl','rock','smsi','gild','cvv','cvx','mtg','rsg','whr','rdi','itt','pid',
                                  'fc','ulh','vgr','dgii','chh','mdt','ttf','amd','bwa','mga','pdco','via','nhs','idti','nfx','pim','susa','dbd','eeft','plt','eml','evp','fcn','nxq','phi','ebay','dmlp','cetv','luv','tess','rdnt','ajg','ft','etr','bse','qgen','glq','ti','vcv','hdb','bce','tues','xcra','bldp','ande','mni','kmb','colb','cni','dst',
                                  'xray','bti','kelya','nick','jva','amwd','exas','eslt','edgw','aph','tkf','csco','dxlg','rtec','has','tsbk','aobc','clf','bax','it','vhi','xrx','hei','utl','srdx','bby','gt','pphm','ezpw','smtc','sfnc','hil','nc','aaww','kfy','idcc','rick','ens','pq','scd','pcmi','odfl','casm','gva','vfc','esv','he','imo','ltc','ug',
                                  'jhx','dcp','vkq','meet','bbgi','enlk','tpl','ntes','oran','egf','wynn','ncr','bzh','dox','dtf','uthr','prph','pag','sbra','cfr','uti','pdt','acco','extr','vz','ago','uis','mgcd','edr','tsi','usat','ttc','flir','bfr','dwdp','cznc','ess','brew','pai','eng','msl','cno','jbl','tol','var','turn','gatx','ottr','dnr','jpc',
                                  'bym','ego','mcf','wor','wsbc','ubfo','wabc','vicl','agu','ctas','idn','tmp','nksh','fds','ftek','manh','audc','prkr','jrjc','fcf','hls','axdx','flo','fhn','egi','jll','tsem','gnw','nxp','stx','brks','lpnt','usap','inva','hyb','rnp','acn','chrw','jcs','lbai','y','son','col','anip','ppih','nwpx','mgee','crvp','fro','matw',
                                  'leco','asrv','muh','caa','plce','ttwo','zf','wmt','mgrc','powl','caty','jta','gaia','duc','sxt','cns','fun','eio','mne','brcd','clct','jdd','ihc','mgf','brk-a','aap','cbak','wpm','rvsb','dave','tkc','prgx','plm','pld','snx','ixys','cbd','rell','dpz','drh','ng','sgma','tlrd','fcnca','azpn','dlx','ensv','fl','gifi','nsu',
                                  'cwco','dis','ksu','gden','acet','tnh','gfy','cgnx','veco','adx','exc','mchp','hhs','dspg','bcv','bak','ihg','npo','entg','idsa','ecf','nxj','cpe','chfc','mac','cdr','ssys','pega','lbtyk','mhk','acu','biib','fred','sho','conn','atu','thrm','mlnk','mgi','ceo','gst','ctbi','ivc','coke','jqc','asfi','tbbk','ntct','ino','mcri',
                                  'jack','ugp','cytr','slg','mrcy','astc','mtr','gmo','oke','gf','tho','ht','phd','renx','scon','amzn','tell','mktx','kep','grmn','rcmt','gern','brs','nflx','uve','achv','efii','blh','aehr','ms','pnc','muj','ubp','itgr','pnk','rpt','mdu','dakt','gpn','avhi','tcbk','mmlp','mdca','fax','acer','cray','sky','amed','sap','nwn',
                                  'abx','bsd','hov','abg','dy','ev','mtn','gbl','iti','call','sgy','usak','snbc','pxlw','prgs','cmu','sqm','syk','aes','dgica','srt','mscc','ba','exr','gass','kmm','wtr','iec','eth','ldr','rdc','algn','ccoi','cto','ufs','rtk','leg','gldd','itus','lts','strt','ssp','gco','iex','wtm','ffbc','jps','jci','amx',])[:N_train]
DEFAULT_VAL_STOCK_LIST = tuple(['ni', 'evj','bhb','amrn','celg','avx','tkr','nymx','ifon','ccc','mvf','mas','cbrl','este','ibkc','sprt','pbct','dynt','vivo','ancx','lii','chnr','aav','twin','mlp','pacw','pui','ads','cbm','pran','sonc','boe','hl','rgr','wyy','thg','lb','ain','roll','gww','db','htgc','wstg','anik','blkb','amswa','expd','dnn','ip','met','spil',
                                'sbux','rnr','ston','orly','awre','nrt','ppg','lh','wcg','eca','re','alt','mli','tst','ttec','ddc','nhi','vrsn','air','htd','mtw','ctrp','actg','tsco','ffic','snv','simo','icon','imax','nee','tjx','fdx','gte','eman','iac','ash','kirk','bbva','apa','stl','tact','mdr','bio','abco','iclr','mco','pmm','cmtl','baf','soho',
                                'sim','mod','scsc','bmrn','tti','opk','bid','hlx','sbs','clx','tpr','egov','sigm','finl','eft','cald','ese','ctsh','aau','apb','syke','mhf','kai','ueic','vcf','swir','ssrm','aep','dgx','hzo','uht','joe','daio','sbny','sify','cnob','nuan','lgi','mini','ca','sjm','wf','sypr','slf','bcr','bbd','trc','cdzi','tcx','glu',
                                'ssw','gncma','pdfs','intx','jhi','bwen','synl','onb','hio','gvp','med','wea','tac','bap','evlv','mur','cbi','nxn','ions','wash','arlp','alg','vvus','lxp','orbk','ecol','relv','hubg','gdv','tat','clro','otex','din','hth','stfc','bobe','aeo','cobz','cik','ivac','sohu','men','cldx','jmm','exac','tfx','lfus','ten','wern','asys',
                                'ato','cck','swm','mfm','jnj','syy','rrgb','jkhy','bfy','smtx','hae','bwp','auto','form','crh','dde','trec','eqt','tm','gcv','wll','rtix','csgp','cgi','acad','exel','kamn','ggb','es','chd','omi','apt','ryaay','pnfp','egp','pgr','at','epd','trk','bch','sskn','pml','unh','bbl','cece','nice','afl','alv','bcpc','uba',
                                'isbc','pzza','bfs','pgnx','town','reg','mck','nwbi','ulti','fnb','vvr','sva','atrc','npv','ptsi','evm','mxf','hckt','jec','mgic','voxx','symc','paas','cers','cnx','smi','aks','ssd','nzf','ngs','wen','vod','gib','frph','mui','spgi','hxl','snhy','prov','paa','bkk','ceva','nwli','cca','klic','mmt','ldl','dswl','bbg','amt',
                                'pht','nvr','osbc','idsy','ztr','trow','vrnt','tvty','bldr','mux','tgp','hst','admp','miy','nca','cvco','ati','arc','ryn','lpx','wft','holx','mue','wvvi','par','mar','newt','rcs','cht','dvd','axe','pfn','sgf','matx','gff','bdge','md','cvgi','are','bll','cbt','phm','cbmx','qcom','dxcm','cnc','boh','mtem','vlo','psdv',
                                'ras','ely','aon','ubnk','pch','cnp','thc','etp','ter','arci','whg','jbss','wsci','hrg','jbht','patk','salm','mmu','ntri','wtfc','akrx','cbk','bhk','cci','usph','fpt','hcn','cib','bvn','ibcp','vtr','icad','mci','googl','evn','rli','avk','eog','laz','immu','agys','cvs','eric','enr','awr','trst','htbk','aet','pdli','msi',
                                'ffin','tmo','arow','urbn','swz','oge','bokf','hix','cig','aey','mfc','lrcx','sdrl','hog','atro','rex','cix','ab','dsx','lorl','tap','nsc','abm','rcl','nem','cpt','nyh','fnsr','mndo','tcf','ew','wmk','afb','emr','fisv','cnty','ghl','ffc','utf','acta','hun','asg','bmi','iivi','ix','cato','gim','siri','dsm','pnm',
                                'ait','pmd','dzsi','grc','mygn','ctws','pbt','cmp','nxst','scvl','schw','cent','igr','axl','kmx','vgm','thff','scco','cli','trmk','osis','anat','uvv','crus','mrln','gpi','hstm','cmco','tdw','cdti','bny','ph','ntz','sbsi','mgu','ir','viav','pico','bkd','iim','hdng','bpfh','pfsw','gel','ben','snmx','deck','cof','mtx','erj',
                                'ttm','asgn','klac','hiho','atge','usna','unm','cvti','sorl','tri','disca','nav','ach','ice','utsi','uslm','lmt','shlm','mbi','trp','wire','sptn','azo','sor','wri','nvec','jnp','pay','srcl','clsn','nbtb','mson','gpl','clfd','hpi','dlb','htld','eia','psa','pes','jof','bmo','ango','rgld','bbby','ffg','nfg','ssi','avdl','ttnp',
                                'antm','cytk','btn','ccbg','sgmo','omcl','chu','tlp','crtn','egle','nnn','lgnd','ctxs','adtn','infy','gol','see','ffiv','sons','erie','dxyn','relx','mhd','bg','mcy','ccj','mza','wfc','vale','iep','mxwl','ndsn','flex','blx','lfc','esnd','phh','eat','osur','muc','iff','txrh','mosy','ams','rdy','itw','bmrc','wat','irbt','ccf',
                                'car','bgt','mxe','myf','incy','bbsi','gpc','aeti','lanc','ipxl','pyn','shbi','nymt','rail','gsk','axti','ainv','phx','ste','cpb','irm','esio','ifn','itri','knl','iag','jhs','agii','hes','tg','isns','rgc','zixi','camp','kof','lee','wbs','rok','sbr','kmt','refr','cat','pnf','krc','tyg','uctt','fcfs','dyn','ups','fms',
                                'eqr','abax','praa','tco','stn','tnp','emitf','snps','kbal','fpp','casi','fgp','rmd','dsw','iba','pjc','cspi','fnhc','nano','moc','gntx','c','ipcc','mtl','agen','itub','gwr','cake','tgt','ntrs','kss','nvg','skyw','atsg','cxh','wdfc','aapl','caj','faro','enlc','vco','mvc','td','spg','strm','ldf','sre','vcel','ffkt','atax',
                                'nmr','layn','cae','seed','csgs','mcs','mdc','dmf','rba','nktr','hmy','evy','esrx','gis','sto','spn','evt','saia','bgcp','coo','del','key','imh','ruth','cop','krny','spe','cgo','orcl','nkx','zbh','wit','prft','gxp','boom','aezs','sgu','cry','irl','skx','cme','wti','snn','kfrc','lnn','aeis','mntx','npk','lxu','rusha',
                                'xntk','cpk','rbpaa','fmbi','crzo','cw','ktf','hmn','hmc','baby','pni','stt','csx','scln','msfg','ofix','ahl','scx','ueps',])[:N_val]

DEFAULT_FULL_SPLIT_STATISTICS = {
    'num_train_timestamps': 7547941,
    'num_val_timestamps': 2904434,
    'num_stocks_train': 2000,
    'num_stocks_val': 770,
}

DEFAULT_INPUT_FEATURES = ('High', 'Open', 'Low', 'Close', 'Volume')
DEFAULT_OUTPUT_FEATURES = (('rise', dict(output_type='categorical', k_next_candle=1)),)

DEFAULT_TRAIN_YEARS = []
DEFAULT_VAL_YEARS = []


class StocksDataset(Dataset):
    time_sample_length: int
    _val_mode: bool
    _stocks_list: (list, tuple)
    config_path: str
    output_features: (list, tuple)
    input_features: (list, tuple)
    stock_name_list: (list, tuple)
    feature_list_with_params: (list, tuple)
    _non_numerical_features: (list, tuple)
    _num_input_features: int
    _num_output_features: int
    years: list
    _relevant_indices: dict

    # TODO: why is this not __init__?
    def init(self, config_path=None, time_sample_length=5,  # features_defaults=None,
             stock_name_list=None, feature_list=None, val_mode=False, output_feature_list=None,
             years=None):

        if config_path is None:
            config_path = 'Shmekel_config.txt'
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.pardir, config_path)
        assert os.path.exists(config_path), 'didnt found file "Shmekel_config.txt", ' \
                                            'please specify the Shmekel config path'

        self.time_sample_length = time_sample_length
        self._val_mode = val_mode
        self._stocks_list = None
        self.config_path = config_path
        self._relevant_indices = {}

        self.output_features = output_feature_list or DEFAULT_OUTPUT_FEATURES
        self.input_features = feature_list or DEFAULT_INPUT_FEATURES

        self.stock_name_list = stock_name_list
        self.stock_name_list = self.stock_name_list or \
                               (DEFAULT_VAL_STOCK_LIST if val_mode else DEFAULT_TRAIN_STOCK_LIST)

        self.feature_list_with_params = [
            x if isinstance(x, (tuple, list)) else (x, {}) for x in self.input_features + self.output_features
        ]
        # for f, params in self.feature_list_with_params:
        #     for key, val in self.features_defaults.items():
        #         f.setdefault(key, val)

        self._non_numerical_features = [('DateTuple', {}), ('RawCandle', {})]

        self._num_input_features = None
        self._num_output_features = None

        self.years = None if years is None else sorted(years)

    def get_default_config(self) -> dict:
        return dict(config_path=None, time_sample_length=1, stock_name_list=None,
                    feature_list=None, val_mode=False, output_feature_list=None)

    @property
    def num_input_features(self):
        if self._num_input_features is None:
            f_with_params = self.feature_list_with_params[:len(self.input_features)]

            self._num_input_features = sum([get_feature(f_name, **params).num_features
                                            for f_name, params in f_with_params])

        return self._num_input_features

    @property
    def num_output_features(self):
        if self._num_output_features is None:
            f_with_params = self.feature_list_with_params[len(self.input_features):]

            self._num_output_features = sum([get_feature(f_name, **params).num_features
                                            for f_name, params in f_with_params])

        return self._num_output_features

    def get_stock_possible_indices(self, s):
        if self.years is None:
            return [i for i in range(len(s) - self.time_sample_length + 1)]

        divided_years_indices_dict = {}
        for i, date_tuple in enumerate(s.not_numerical_feature_list[0]):
            divided_years_indices_dict.setdefault(date_tuple[0], []).append(i)

        last_year = -2
        groups = []
        for year in self.years:
            if year == last_year + 1:
                groups[-1].extend(divided_years_indices_dict.get(year, []))
            else:
                groups.append(divided_years_indices_dict.get(year, []))
            last_year = year

        indices_groups = []
        for group_indices in groups:
            if (len(group_indices) - self.time_sample_length + 1) > 0:
                indices_groups.append(group_indices[:-self.time_sample_length + 1])

        return sum(indices_groups, [])

    def stock_effective_len(self, s):
        if s.stock_tckt not in self._relevant_indices:
            self._relevant_indices[s.stock_tckt] = self.get_stock_possible_indices(s)

        return len(self._relevant_indices[s.stock_tckt])

    def stock_and_local_index_from_global_index(self, index):
        stock = None
        for _stock in self.stocks_list:
            if index < self.stock_effective_len(_stock):
                stock = _stock
                break
            index = index - self.stock_effective_len(_stock)

        if stock is None:
            raise IndexError

        return self._relevant_indices[stock.stock_tckt][index], stock

    def __getitem__(self, o_index) -> dict:
        index, stock = self.stock_and_local_index_from_global_index(o_index)

        inputs = copy(stock.feature_matrix[index: index + self.time_sample_length, :self.num_input_features])
        outputs = copy(stock.feature_matrix[index + self.time_sample_length - 1, self.num_input_features:])

        item = {'inputs': inputs, 'outputs': outputs, 'id': o_index, 'stock': stock, '_id': index}
        for (s, _), f in zip(self._non_numerical_features, stock.not_numerical_feature_list):
            item[s] = f[index + self.time_sample_length - 1]

        return item

    @property
    def stocks_list(self):
        if self._stocks_list is None:
            reader = DataReader(self.config_path)

            logger.info('Loading Stocks:')
            self._stocks_list = []
            for tckt in tqdm(self.stock_name_list):
                self._stocks_list.append(Stock(tckt, reader.load_stock(tckt), feature_list=[get_feature(f_name, **params)
                              for f_name, params in self.feature_list_with_params + self._non_numerical_features]))
        return self._stocks_list

    @property
    def val_mode(self) -> bool:
        return self._val_mode

    @property
    def input_shape(self) -> tuple:
        return self.time_sample_length, self.num_input_features

    @property
    def output_shape(self) -> tuple:
        return tuple([self.num_output_features])

    def __len__(self) -> int:
        return sum([self.stock_effective_len(stock) for stock in self.stocks_list])

    def __str__(self) -> str:
        return 'StocksDataSet-' + 'val' * self.val_mode

    def __bool__(self) -> bool:
        return bool(self.stock_name_list)
