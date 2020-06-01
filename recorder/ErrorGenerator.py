#this program reads hand_manifold.csv and generates 
#a data set of abnormal hand poses by bending fingers beyond their bounds

#generate array of 45 booleans
from itertools import combinations_with_replacement
from sympy.utilities.iterables import multiset_permutations
import copy
import numpy as np
import csv

#get input
data = []
with open("hand_manifold.csv") as file:
    reader = list(csv.reader(file))
    for i in range(len(reader)):
        data.append([])
        for j in range(len(reader[0])):
            data[i].append(float(reader[i][j]))

#test input
test_set = [
    [18.429660390871337,9.220553981947212,42.094206449342465,7.881703966816644e-06,-0.30567432995978727,32.0110324774342,3.260652263747943e-07,-10.4789612403691,19.314142510558547,-2.312024293103171,-4.015887785467509,40.070516354405925,-2.14903962536539e-06,3.791033418326495,22.374791909686284,2.623769744580784e-06,2.2182048005394095,15.887632248729776,-9.603706900546294,-7.594047869075142,43.567914669172694,2.985734553817565e-06,4.571818113218988,26.304704506882036,-3.836729996820054e-06,2.554943399204059,17.457917847747662,-6.722213151969644,-9.657893950932584,40.26570660221307,-1.6425365121364166e-06,3.4531354763947952,25.779269402105577,9.233418074572342e-08,2.007298580779373,17.427259389089034,-14.640641341820261,-7.929117302026165,28.72189044740131,1.5864170523904875e-06,2.7917277531730162,18.150388162905987,2.775684135869483e-07,2.125998003844902,16.04344779511809],
    [18.397483141000166,9.2063009262115,42.11139759048937,8.137835088461998e-06,-0.27692402897493906,32.01129569313166,-3.817106647652224e-06,-10.439459768519857,19.335526185536835,-2.3278017943755636,-3.990895883479957,40.07209919055104,2.189931718632465e-06,3.794437838411442,22.37421451565355,-4.892853275428877e-06,2.219777516092508,15.887413750907076,-9.649664300504844,-7.507681414711238,43.572730242265244,-3.369753791560015e-07,4.584738510431185,26.30245820749792,-2.9998845620937686e-06,2.560666957020336,17.457074578949687,-6.732029291341233,-9.634191835643636,40.26973821597393,5.152040740696862e-07,3.4873267192594337,25.77466703181433,-5.659874338093118e-07,2.0268225578719195,17.4249984489372,-14.641404073079897,-7.932586509200142,28.720543590279533,4.214610962449683e-06,2.794467725044016,18.149965727980426,-1.5669076494084777e-06,2.1281532438211364,16.043165889450176],
    [18.348599509938715,9.125509393948608,42.15029540912879,-3.071015774125385e-06,-0.2503615520852005,32.01151073993517,-3.0050831556494018e-06,-10.392677528296336,19.360714103124074,-2.3705524607987236,-3.914763025111361,40.07710375351775,-2.710238797831721e-06,3.7903511772353666,22.374907126800906,1.1434751758798711e-06,2.215996701851417,15.88793951516125,-9.695729112257725,-7.440850871896714,43.57396714398592,1.84409165360222e-06,4.598762469772902,26.30000739050329,3.442841389755813e-07,2.5673244079834423,17.456097017744458,-6.745685193300764,-9.615606523713684,40.27189885746936,3.0099787515780463e-06,3.5268234703039814,25.76929172723108,-2.9950184670113345e-06,2.049466001095065,17.42234939227157,-14.644738644294806,-7.941498182532603,28.716380390630267,2.241837973482319e-06,2.808172381828655,18.14785174731525,-8.666634743192958e-07,2.1387984006374157,16.04174874590732],
    [18.316247380292527,9.079669873644471,42.17425893173947,1.1376554952313711e-05,-0.25287177731536303,32.01149267209442,-5.754923242839993e-06,-10.391468346033164,19.361358585044947,-2.4057783187789603,-3.891833232615615,40.07723491283827,-1.1938286856860714e-06,3.7910152462599633,22.374794240810903,1.2356855645379028e-06,2.2159804507598193,15.887942610881577,-9.701170354834057,-7.439796378556802,43.572935679969845,-3.7655724698382187e-06,4.606361664852612,26.298679056282225,1.6223408385585003e-06,2.571523358585381,17.455479561757418,-6.755079570843083,-9.596452343105984,40.27489148790847,1.8486319710575572e-06,3.5504718814101306,25.76604401187134,3.719246789213315e-06,2.0628774030479513,17.42076551361953,-14.653395775557716,-7.932869275122412,28.7143502594313,4.536590390458173e-06,2.8286335441527255,18.144672367955202,2.4050421747467965e-06,2.154210570444246,16.039682771801246],
    [18.284923254403687,9.033200314465116,42.19782521553306,-1.4084294825522647e-05,-0.24744971932219073,32.01153282819662,4.448542319401838e-06,-10.373191654440141,19.371160771788198,-2.892858484743119,-3.7536900527909243,40.05821105330086,-1.2741397359405937e-06,3.8467595631696856,22.36527606794971,-2.5975421813484445e-08,2.24611930279374,15.883712508105916,-9.627162072583953,-7.378127271793106,43.59982760204672,-4.33702611601916e-08,4.67006351267794,26.28744044409723,8.524784766983373e-07,2.6059638745061378,17.450374101776102,-6.704916498164156,-9.471518211181705,40.31283003812461,-2.6058488398206237e-06,3.663253250053728,25.750254068601933,1.8677170263714515e-06,2.126279310162528,17.41314458106846,-14.661518367754754,-7.841116815457825,28.73539768569436,4.1374422510287445e-06,2.8840650156506964,18.135944347916155,-2.949751056302574e-06,2.194484727061214,16.034224766200424],
    [18.282613660027998,8.98906697262477,42.208241659679935,1.576540728187581e-06,-0.2702680010163636,32.01135291049862,-1.135702238919123e-06,-10.433470543064129,19.338755405065754,-3.714754005897394,-3.6091921838720236,40.003665662759886,1.2147719505861687e-06,3.8473589142159055,22.365174491806748,3.0528008938546236e-06,2.2438630918291054,15.884030104560274,-9.449492050411006,-7.257001195117345,43.65898525499877,-5.759897989676688e-07,4.7372650970987475,26.275416600255102,-4.817762966347061e-07,2.641331063764503,17.445055061766773,-6.637212002174019,-9.325966197210874,40.35794435175418,-3.092143440852624e-06,3.706983594679249,25.743992118335726,2.00618840962008e-06,2.14937266796265,17.41030985660901,-14.637586559256725,-7.709656837209233,28.783129704051714,4.467432623833645e-07,2.940093946366218,18.126946101774532,1.5174810132911887e-06,2.23436568917102,16.028715987885725],
    [18.259611244146072,8.89361639160427,42.23841639184093,1.641633705418144e-06,-0.2969950588441428,32.01111357235619,-2.521303386515683e-06,-10.47343508714389,19.3171428531012,-5.025589279506679,-3.474727316912164,39.872155656053835,-3.466503484261807e-06,3.8569643675187595,22.363518803812205,2.0954435262510174e-06,2.2471578268277597,15.883563418529178,-9.063597930785102,-7.240892379129943,43.74339707668102,4.095453951968864e-06,4.793725596937892,26.265170259932514,-2.686977600419027e-07,2.6726635364078914,17.440283658732376,-6.499323165065521,-9.249613943765711,40.39794104017488,-2.6015445104832224e-06,3.710090753687922,25.743543871915193,-1.1530706522933087e-06,2.15003059042856,17.41022515003588,-14.605942979103183,-7.6613626517683056,28.8120836904866,-7.193302735686302e-06,2.9444164638257178,18.126248289654104,6.717370971465186e-06,2.236630984415639,16.02839642797887],
    [18.23998483850226,8.87517447479551,42.25077035957004,-1.0592119590313587e-06,-0.3060134678908941,32.011026515800324,6.529629986928853e-06,-10.478037485022027,19.314650043987207,-5.6809501165780105,-3.420106246276199,39.788801480182855,-1.1671865447482332e-06,3.852532070237854,22.364282037120226,3.0257951415535445e-06,2.2436114553532125,15.884064598993309,-8.91094653530922,-7.24751259269488,43.773650527800854,-4.5241973190002227e-07,4.801775112086939,26.263701950781705,2.3311743317933065e-06,2.6772379894117853,17.4395764191255,-6.379033712445581,-9.158984763774841,40.437740167225385,2.3465697092106552e-06,3.747693335288007,25.738095763941345,1.3141225068125095e-06,2.1703541588837316,17.407706761367955,-14.53445051378719,-7.643680765030709,28.852906140761156,2.9093374718236475e-06,2.957294663592495,18.12414964127634,1.2502023079008495e-06,2.2460676954978958,16.02707918054461],
    [18.207323222811496,8.868901938049785,42.26617144761521,6.396740253933331e-06,-0.300978812785484,32.01107695339016,-2.317927336292769e-06,-10.475973149679884,19.31576958831242,-6.8320412503108265,-3.3820255825184438,39.610657105460845,3.839700237762145e-07,3.9267606051330946,22.35136888497299,-1.3447134561417329e-06,2.286162896849387,15.87799659501683,-8.77847481663878,-7.29013092869846,43.79334055280534,4.819123882171539e-07,4.855756075999287,26.253771323286895,4.478162338727998e-07,2.7082192317871057,17.43479807580922,-6.0715172966607245,-9.077927000147602,40.5033059882613,-2.1601766739820505e-06,3.860653399546919,25.721398233863233,1.1855313180042515e-06,2.234355480205113,17.3996050567556,-14.207966296635608,-7.588098218326651,29.029655113139803,3.0555757124517413e-06,3.0381248197780435,18.110772576069465,-2.0917664655950574e-06,2.3063981810966157,16.01850950826963],
    [18.141175012389144,8.83416618942874,42.30187340409019,3.778618357763719e-06,-0.2543481579067546,32.01147922022324,-3.8748541761890465e-06,-10.53249525435146,19.285009190282395,-8.372007371984939,-3.2299772012514856,39.32678175248045,4.695475954719086e-06,4.05862117588929,22.3278015862251,-1.8445270200118102e-08,2.360197690438842,15.867160340760872,-8.634555841647664,-7.26197615576596,43.82661384802644,1.8381532189692962e-06,4.977456758346003,26.230975368114443,-2.1319107688100303e-06,2.7758008284691416,17.42416630367401,-5.684689466195717,-9.02734513241036,40.570692846555175,1.776532711161849e-06,4.056485172257419,25.691241219397924,2.016170082441704e-06,2.3470864248618493,17.384754690820998,-13.266057571996168,-7.425062067855239,29.513497431553226,3.2242068113141897e-06,3.220151498362869,18.07929831929591,-6.680924080626482e-07,2.4412549498334712,15.998510404498715],
    [18.08207718528162,8.804597629574621,42.33332935574975,4.039818724876909e-06,-0.17168004619689192,32.01203272623947,-6.596061404628983e-06,-10.550159883375427,19.275343459475625,-9.036889909946762,-3.1972708772006877,39.182028866264936,-4.738774652679467e-07,4.122494966795026,22.31609842067008,-1.5111952258450856e-06,2.3968648295597115,15.861662454195978,-8.698109256127196,-7.2330234377996305,43.81883552220608,1.2972700265123649e-06,5.030290620914028,26.220891840311737,9.513631145097179e-07,2.804803126175463,17.419518333851535,-5.386689536163418,-8.944986339629963,40.62955690782337,-2.1458504511073784e-06,4.177128266597964,25.671902015815043,4.0306537831824585e-06,2.4156114680923677,17.37536478318699,-12.23386921644343,-7.218111531260144,30.006625925328503,3.746238601110008e-06,3.3591545246874315,18.053985690465566,-3.980280994353436e-06,2.5418109546752468,15.982849999799637],
    [17.88537720100519,8.747473476389882,42.42862351127442,-3.768841448703597e-06,0.1334599935147427,32.01221335073025,-2.919560507130825e-06,-10.48517183484555,19.310772036955132,-9.77507239712617,-3.112786644883644,39.01124986173125,-1.2047210358190341e-06,4.230925393546826,22.295793544425774,3.86303110389008e-07,2.4585048322867653,15.852226019543226,-9.394020859475575,-7.050128465715138,43.704828431695766,-3.1550535135593805e-06,5.13867703043988,26.19987016623714,-2.0625878427438238e-07,2.86186297368047,17.41023501692696,-4.682994458000513,-8.66504108240273,40.777158725874116,4.2777687614403703e-07,4.286935626079401,25.65379183254883,-1.564610781379372e-06,2.4740644781201477,17.367142673977938,-9.574582094171657,-6.801887042083431,31.05202055404891,2.164189717102083e-06,3.366945345817981,18.052535258251535,-1.3547008611003442e-06,2.5377532438664154,15.983488392846708],
    [17.705388476679126,8.884422678076572,42.475636185383955,4.1869580158504505e-07,0.45042674974197716,32.009322890716135,8.581882690705811e-06,-10.501452465341565,19.30192444338065,-10.133366858523267,-3.0562538537140034,38.92420098682492,-1.3102467217862568e-06,4.2784439557830565,22.28672601069708,4.0172744064648214e-07,2.485056565147911,15.848082579463146,-10.001725930716479,-6.899035427110089,43.593954996045895,-3.012061504215069e-06,5.211714129852548,26.1854376615364,-2.360332400641596e-07,2.899554935029779,17.40399904235133,-4.202180092888728,-8.560563156364775,40.85154352010095,-1.108286627804489e-06,4.314219604460238,25.6492196493839,2.1615416052611636e-06,2.4879375594373228,17.365156950359793,-7.8050510684735785,-6.7158022744488655,31.561772168369338,6.028405754476829e-06,3.3987778872686905,18.046570238128837,-1.3080977173673958e-06,2.560056291253208,15.97992855923663],
    [17.575200976793006,9.116747451555746,42.48047074073126,-5.035256919860842e-06,0.8307219655741438,32.0017105248852,5.529118143776657e-06,-10.511171876089902,19.296635822650728,-10.51446012480498,-2.901338194204396,38.834871435286274,-7.275310283105796e-07,4.2901956914563515,22.284468234795465,1.634349662715806e-06,2.4890084291251915,15.847464843331412,-10.40497926122298,-6.646231924177622,43.53881274231803,5.007150107871894e-07,5.204084665487467,26.186953614419924,-3.340176485622237e-06,2.890320569980152,17.405538402108647,-3.7838947586144656,-8.732292480906098,40.85608024254339,1.5211464283737541e-06,4.4422376907917664,25.627353683421333,-1.928153460450943e-06,2.5654159215514882,17.353887767521393,-5.633701919370852,-6.712462477082074,32.02140781636146,7.219782507394967e-06,3.5356031190675132,18.020262795340642,-6.5329329287378e-06,2.6633498733592367,15.963045279365874],
    [17.495538535484627,9.232406785604514,42.48837669528295,8.568588543589328e-06,1.0082652720512417,31.996608936354427,-2.287667682843164e-06,-10.542398928062738,19.279589332917297,-10.797446751539535,-2.7560640761329442,38.76774608051781,-7.065137763717644e-07,4.262304899241231,22.28981929254078,8.764008674333468e-07,2.470074071461884,15.850424362043356,-10.62780904609011,-6.315700187855115,43.53418982006933,1.932175043073414e-06,5.112491228253065,26.204988428904613,-4.719546892850701e-06,2.8333342935384866,17.414903906016114,-3.2346284946574784,-8.214601068381409,41.01033241110816,-1.7763082489352655e-06,4.360477053251374,25.64139591203308,1.780147425023415e-06,2.5086357467683573,17.362177975091353,-2.657092924374112,-6.216849286391112,32.503181853969565,-5.304347956425204e-07,3.5590711446269863,18.015641773473508,1.3353108012648107e-06,2.669116011139633,15.962076683258367]
]

# List positive and negative out of bounds values (5 units beyond largest/smallest recorded values)
positive_break = [
    26.0,
    35.0,
    50.0,

    5.0,
    29.0,
    37.0,

    5.0,
    29.0,
    27.0,

    19.0,
    45.0,
    47.0,

    5.0,
    27.0,
    29.0,

    5.0,
    27.0,
    22.0,

    18.0,
    47.0,
    49.0,

    5.0,
    32.0,
    33.0,

    5.0,
    22.0,
    23.0,

    13.0,
    41.0,
    49.0,

    5.0,
    31.0,
    33.0,

    5.0,
    21.0,
    23.0,

    15.0,
    40.0,
    37.0,

    5.0,
    23.0,
    23.0,
    
    5.0,
    18.0,
    14.0
]

negative_break = [
    -1.0,
    -30.0,
    28.0,

    -5.0,
    -14.0,
    17.0,

    -5.0,
    -18.0,
    -1.0,

    -19.0,
    -27.0,
    3.0,

    -5.0,
    -4.0,
    -5.0,

    -5.0,
    -5.0,
    -5.0,

    -18.0,
    -30.0,
    2.0,
    
    -5.0,
    -3.0,
    -3.0,

    -5.0,
    -5.0,
    -2.0,

    -18.0,
    -27.0,
    2.0,

    -5.0,
    -5.0,
    -3.0,

    -5.0,
    -5.0,
    0.0,

    -19.0,
    -23.0,
    -1.0,

    -5.0,
    -5.0,
    -5.0,

    -5.0,
    -5.0,
    11.0
]

mult = [1,1,1,1]

#make Swan Neck Deformity on row
def swan_neck(input, matrix_row, random_row):
    if(matrix_row[0] == 1):
        input[12] = 0 + random_row[12]
        input[13] = negative_break[13] + random_row[13]
        input[14] = 0 + random_row[14]
        input[15] = 0 + random_row[15]
        input[16] = positive_break[16] + random_row[16]
        input[17] = 0 + random_row[17]
    if(matrix_row[1] == 1):
        input[21] = 0 + random_row[21]
        input[22] = negative_break[22] + random_row[22]
        input[23] = 0 + random_row[23]
        input[24] = 0 + random_row[24]
        input[25] = positive_break[25] + random_row[25]
        input[26] = 0 + random_row[26]
    if(matrix_row[2] == 1):
        input[30] = 0 + random_row[30]
        input[31] = negative_break[31] + random_row[31]
        input[32] = 0 + random_row[32]
        input[33] = 0 + random_row[33]
        input[34] = positive_break[34] + random_row[34]
        input[35] = 0 + random_row[35]
    if(matrix_row[3] == 1):
        input[39] = 0 + random_row[39]
        input[40] = negative_break[40] + random_row[40]
        input[41] = 0 + random_row[41]
        input[42] = 0 + random_row[42]
        input[43] = positive_break[43] + random_row[43]
        input[44] = 0 + random_row[44]

#make Boutonniere Deformity on row
def boutonniere(input, matrix_row, random_row):
    if(matrix_row[0] == 1):
        input[12] = 0 + random_row[12]
        input[13] = positive_break[13] + random_row[13]
        input[14] = 0 + random_row[14]
        input[15] = 0 + random_row[15]
        input[16] = negative_break[16] + random_row[16]
        input[17] = 0 + random_row[17]
    if(matrix_row[1] == 1):
        input[21] = 0 + random_row[21]
        input[22] = positive_break[22] + random_row[22]
        input[23] = 0 + random_row[23]
        input[24] = 0 + random_row[24]
        input[25] = negative_break[25] + random_row[25]
        input[26] = 0 + random_row[26]
    if(matrix_row[2] == 1):
        input[30] = 0 + random_row[30]
        input[31] = positive_break[31] + random_row[31]
        input[32] = 0 + random_row[32]
        input[33] = 0 + random_row[33]
        input[34] = negative_break[34] + random_row[34]
        input[35] = 0 + random_row[35]
    if(matrix_row[3] == 1):
        input[39] = 0 + random_row[39]
        input[40] = positive_break[40] + random_row[40]
        input[41] = 0 + random_row[41]
        input[42] = 0 + random_row[42]
        input[43] = negative_break[43] + random_row[43]
        input[44] = 0 + random_row[44]

def hitchhiker(input, random_row):
    input[3] = 0 + random_row[3]
    input[4] = 24.0 + random_row[4]
    input[5] = 0 + random_row[5]
    input[6] = 0 + random_row[6]
    input[7] = -13.0 + random_row[7]
    input[8] = 0 + random_row[8]

#pass a set of replacement values and an input set
def error_generator(replacements, input):
    comb = list(combinations_with_replacement([1,0], 4))
    random_matrix = np.random.randn(len(input), len(input[0]))
    # Generate a set of every possible combination of 1's and 0's of length equal
    # to the multipliers set, leaving the pure 0 set out
    boolean_matrix = []
    for i in range(len(comb)):
        boolean_row = list(multiset_permutations(comb[i]))
        if 1 in boolean_row[0]:
            boolean_matrix += boolean_row
    
    swan_set = copy.deepcopy(input)
    boutonniere_set = copy.deepcopy(input)
    hitchhiker_set = copy.deepcopy(input)
    bool_iter = 0
    for i in range(len(input)):
        swan_neck(swan_set[i], boolean_matrix[bool_iter], random_matrix[i])
        with open('swan_neck.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(swan_set[i])
            file.close()

        boutonniere(boutonniere_set[i], boolean_matrix[bool_iter], random_matrix[i])
        with open('boutonniere.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(boutonniere_set[i])
            file.close()

        bool_iter += 1
        if bool_iter >= len(boolean_matrix):
            bool_iter = 0
        
        hitchhiker(hitchhiker_set[i], random_matrix[i]) 
        with open('hitchhiker.csv', 'ab') as file:
            writer = csv.writer(file)
            writer.writerow(hitchhiker_set[i])
            file.close()
    
    #print(swan_set)
    #print(boutonniere_set)
    #print(hitchhiker_set)
    #print(random_matrix)



error_generator(mult, data)