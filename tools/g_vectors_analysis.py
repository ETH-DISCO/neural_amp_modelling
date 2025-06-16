import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Your provided JSON data as a string
json_data = """
[
    {
                "g_vector": [
                    0.06730923056602478,
                    0.9866757988929749,
                    0.8271890878677368,
                    0.012945777736604214,
                    0.7413782477378845,
                    0.5684248208999634
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0000.wav"
            },
            {
                "g_vector": [
                    0.02512281946837902,
                    0.9386854767799377,
                    0.8919239640235901,
                    0.06630312651395798,
                    0.734160304069519,
                    0.4880954623222351
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0001.wav"
            },
            {
                "g_vector": [
                    0.16165465116500854,
                    0.983393132686615,
                    0.8206357359886169,
                    0.011423355899751186,
                    0.6915324926376343,
                    0.5768606066703796
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0002.wav"
            },
            {
                "g_vector": [
                    0.03747908025979996,
                    0.9931419491767883,
                    0.11053735017776489,
                    0.15667004883289337,
                    0.7402426600456238,
                    0.026030046865344048
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0003.wav"
            },
            {
                "g_vector": [
                    0.05141374468803406,
                    0.01064390130341053,
                    0.8271881937980652,
                    0.2851499021053314,
                    0.7527125477790833,
                    0.012992950156331062
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0004.wav"
            },
            {
                "g_vector": [
                    0.53226584,
                    0.8646086,
                    0.4145501,
                    0.06372362,
                    0.74984103,
                    0.20369919
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0010.wav"
            },
            {
                "g_vector": [
                    0.43941462,
                    0.88550663,
                    0.00869179,
                    0.5813503,
                    0.67658406,
                    0.12426575
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0011.wav"
            },
            {
                "g_vector": [
                    0.6453688,
                    0.02586384,
                    0.29719195,
                    0.22350264,
                    0.6086145,
                    0.02717558
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0012.wav"
            },
            {
                "g_vector": [
                    0.49073732,
                    0.5301111,
                    0.7270142,
                    0.49153736,
                    0.9144966,
                    0.00385426
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0013.wav"
            },
            {
                "g_vector": [
                    0.6900244355201721,
                    0.8157644867897034,
                    0.1652367264032364,
                    0.7214745879173279,
                    0.496310830116272,
                    0.10274024307727814
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0020.wav"
            },
            {
                "g_vector": [
                    0.861016035079956,
                    0.07780534774065018,
                    0.9926319122314453,
                    0.20138491690158844,
                    0.8446781635284424,
                    0.3455277383327484
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0021.wav"
            },
            {
                "g_vector": [
                    0.11200813204050064,
                    0.9668230414390564,
                    0.9916048049926758,
                    0.8242881894111633,
                    0.9663863778114319,
                    0.011306433007121086
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0022.wav"
            },
            {
                "g_vector": [
                    0.371713250875473,
                    0.13305236399173737,
                    0.7833674550056458,
                    0.018169088289141655,
                    0.5964839458465576,
                    0.031446296721696854
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0023.wav"
            },
            {
                "g_vector": [
                    0.006853445898741484,
                    0.007380574010312557,
                    0.9434449672698975,
                    0.9655570387840271,
                    0.9932181239128113,
                    0.9782132506370544
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0024.wav"
            },
            {
                "g_vector": [
                    0.49702224135398865,
                    0.118361696600914,
                    0.3896504342556,
                    0.6456631422042847,
                    0.9467802047729492,
                    0.2002968043088913
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0030.wav"
            },
            {
                "g_vector": [
                    0.9775493144989014,
                    0.8134959936141968,
                    0.14690999686717987,
                    0.00851618591696024,
                    0.7508080005645752,
                    0.9457495808601379
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0031.wav"
            },
            {
                "g_vector": [
                    0.9765564203262329,
                    0.04662007465958595,
                    0.423391729593277,
                    0.1496831625699997,
                    0.6128465533256531,
                    0.5582066774368286
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0032.wav"
            },
            {
                "g_vector": [
                    0.23012307286262512,
                    0.6634036898612976,
                    0.2359635978937149,
                    0.04243272542953491,
                    0.982045590877533,
                    0.006430954206734896
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0033.wav"
            },
            {
                "g_vector": [
                    0.432015985250473,
                    0.003403332084417343,
                    0.9982287287712097,
                    0.8396477103233337,
                    0.8926090002059937,
                    0.8176581263542175
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0040.wav"
            },
            {
                "g_vector": [
                    0.98102205991745,
                    0.9884305000305176,
                    0.5954511761665344,
                    0.4114452302455902,
                    0.48883238434791565,
                    0.38870394229888916
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0041.wav"
            },
            {
                "g_vector": [
                    0.7924257516860962,
                    0.978693962097168,
                    0.40713608264923096,
                    0.09860719740390778,
                    0.5055555105209351,
                    0.989423930644989
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0042.wav"
            },
            {
                "g_vector": [
                    0.4411157965660095,
                    0.09802728146314621,
                    0.988312304019928,
                    0.11016533523797989,
                    0.8429800868034363,
                    0.07711248844861984
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0043.wav"
            },
            {
                "g_vector": [
                    0.9734296798706055,
                    0.17675015330314636,
                    0.660631537437439,
                    0.0038098928052932024,
                    0.251407265663147,
                    0.5189512372016907
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0050.wav"
            },
            {
                "g_vector": [
                    0.9736301898956299,
                    0.8318794965744019,
                    0.9787324070930481,
                    0.4002455770969391,
                    0.8502886891365051,
                    0.36576345562934875
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0051.wav"
            },
            {
                "g_vector": [
                    0.19123442471027374,
                    0.6152593493461609,
                    0.02412368729710579,
                    0.008538047783076763,
                    0.9928164482116699,
                    0.9186423420906067
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0052.wav"
            },
            {
                "g_vector": [
                    0.04862736538052559,
                    0.9826229810714722,
                    0.9862235188484192,
                    0.9686102271080017,
                    0.39663076400756836,
                    0.9383262991905212
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0053.wav"
            },
            {
                "g_vector": [
                    0.9421733021736145,
                    0.5798728466033936,
                    0.9975504279136658,
                    0.07493770867586136,
                    0.5063042640686035,
                    0.85174161195755
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0060.wav"
            },
            {
                "g_vector": [
                    0.9750158190727234,
                    0.9152981638908386,
                    0.9943996071815491,
                    0.11052674055099487,
                    0.596583902835846,
                    0.6070519685745239
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0061.wav"
            },
            {
                "g_vector": [
                    0.7409691214561462,
                    0.2974367141723633,
                    0.9930872321128845,
                    0.510461151599884,
                    0.5878965854644775,
                    0.024255476891994476
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0062.wav"
            },
            {
                "g_vector": [
                    0.38086700439453125,
                    0.2654130458831787,
                    0.7954410314559937,
                    0.9624946713447571,
                    0.7241172194480896,
                    0.9907999634742737
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0063.wav"
            },
            {
                "g_vector": [
                    0.8684354424476624,
                    0.9580699801445007,
                    0.9435182213783264,
                    0.021426010876893997,
                    0.34738853573799133,
                    0.04148231819272041
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0064.wav"
            },
            {
                "g_vector": [
                    0.007589069195091724,
                    0.04633757472038269,
                    0.9872246980667114,
                    0.5129563212394714,
                    0.48973309993743896,
                    0.4146553874015808
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0065.wav"
            },
            {
                "g_vector": [
                    0.25069794058799744,
                    0.3230625092983246,
                    0.9509857296943665,
                    0.9956609606742859,
                    0.9933627843856812,
                    0.7275301218032837
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0070.wav"
            },
            {
                "g_vector": [
                    0.9355644583702087,
                    0.7060459852218628,
                    0.867233395576477,
                    0.008342036046087742,
                    0.5276060104370117,
                    0.008593915030360222
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0071.wav"
            },
            {
                "g_vector": [
                    0.3305109143257141,
                    0.25647905468940735,
                    0.04613930359482765,
                    0.7053649425506592,
                    0.9895293116569519,
                    0.09219320118427277
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0072.wav"
            },
            {
                "g_vector": [
                    0.974435567855835,
                    0.7835215926170349,
                    0.7343708872795105,
                    0.9770879745483398,
                    0.9816934466362,
                    0.9861342310905457
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0080.wav"
            },
            {
                "g_vector": [
                    0.9396310448646545,
                    0.8422842621803284,
                    0.045262012630701065,
                    0.01138137187808752,
                    0.44151079654693604,
                    0.020038722082972527
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0081.wav"
            },
            {
                "g_vector": [
                    0.8629069924354553,
                    0.16234780848026276,
                    0.9949413537979126,
                    0.22703130543231964,
                    0.3517918586730957,
                    0.42196500301361084
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0082.wav"
            },
            {
                "g_vector": [
                    0.10713706165552139,
                    0.04954638332128525,
                    0.10316578298807144,
                    0.9921030402183533,
                    0.9792287349700928,
                    0.44214195013046265
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0083.wav"
            },
            {
                "g_vector": [
                    0.9819595217704773,
                    0.9932859539985657,
                    0.012797396630048752,
                    0.9839800596237183,
                    0.955049991607666,
                    0.9913820624351501
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0090.wav"
            },
            {
                "g_vector": [
                    0.9922814965248108,
                    0.9953086972236633,
                    0.020839011296629906,
                    0.5332239270210266,
                    0.8652929663658142,
                    0.012445404194295406
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0091.wav"
            },
            {
                "g_vector": [
                    0.5554093718528748,
                    0.9888699054718018,
                    0.9898842573165894,
                    0.023653006181120872,
                    0.9883941411972046,
                    0.9447124004364014
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0092.wav"
            },
            {
                "g_vector": [
                    0.9908267855644226,
                    0.5572497248649597,
                    0.9765452742576599,
                    0.46605661511421204,
                    0.43917104601860046,
                    0.013842307962477207
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0093.wav"
            },
            {
                "g_vector": [
                    0.18809661269187927,
                    0.022964773699641228,
                    0.9760611057281494,
                    0.9476643204689026,
                    0.855381965637207,
                    0.9863309860229492
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0094.wav"
            },
            {
                "g_vector": [
                    0.9835652709007263,
                    0.9924290776252747,
                    0.01613580994307995,
                    0.3757149577140808,
                    0.9711751341819763,
                    0.9971948862075806
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0100.wav"
            },
            {
                "g_vector": [
                    0.9887827038764954,
                    0.9509878754615784,
                    0.4670525789260864,
                    0.9512001872062683,
                    0.8037012815475464,
                    0.006768971681594849
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0101.wav"
            },
            {
                "g_vector": [
                    0.9440052509307861,
                    0.9932469129562378,
                    0.969034731388092,
                    0.9756191968917847,
                    0.9610981941223145,
                    0.189816415309906
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0102.wav"
            },
            {
                "g_vector": [
                    0.8687502145767212,
                    0.994525671005249,
                    0.9917333126068115,
                    0.9162574410438538,
                    0.9737228751182556,
                    0.8275917172431946
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0103.wav"
            },
            {
                "g_vector": [
                    0.8633626699447632,
                    0.9935426712036133,
                    0.9977293610572815,
                    0.8902521133422852,
                    0.9746222496032715,
                    0.952171802520752
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0104.wav"
            },
            {
                "g_vector": [
                    0.10184600204229355,
                    0.6790528297424316,
                    0.9826307892799377,
                    0.9867419600486755,
                    0.9858478307723999,
                    0.9857359528541565
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0105.wav"
            },
            {
                "g_vector": [
                    0.9816546440124512,
                    0.0023078969679772854,
                    0.6955685615539551,
                    0.6606557965278625,
                    0.5492286086082458,
                    0.9538277387619019
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0106.wav"
            },
            {
                "g_vector": [
                    0.17360593378543854,
                    0.20098905265331268,
                    0.39401063323020935,
                    0.7453489899635315,
                    0.9769944548606873,
                    0.04575669765472412
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0107.wav"
            },
            {
                "g_vector": [
                    0.9506881833076477,
                    0.9325565099716187,
                    0.9846435785293579,
                    0.2510126531124115,
                    0.016669392585754395,
                    0.038522493094205856
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0108.wav"
            },
            {
                "g_vector": [
                    0.6484142541885376,
                    0.3758460283279419,
                    0.9757989645004272,
                    0.3627309203147888,
                    0.4656676650047302,
                    0.024681225419044495
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0109.wav"
            },
            {
                "g_vector": [
                    0.9892542362213135,
                    0.0022626551799476147,
                    0.9419844746589661,
                    0.9906215667724609,
                    0.9893350005149841,
                    0.985572874546051
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0110.wav"
            },
            {
                "g_vector": [
                    0.9892266988754272,
                    0.9878836274147034,
                    0.9937301874160767,
                    0.2537764012813568,
                    0.9889084696769714,
                    0.9651358723640442
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0111.wav"
            },
            {
                "g_vector": [
                    0.9874176979064941,
                    0.9953533411026001,
                    0.21340596675872803,
                    0.5040692687034607,
                    0.6793172359466553,
                    0.012758779339492321
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0112.wav"
            },
            {
                "g_vector": [
                    0.31243789196014404,
                    0.007019761484116316,
                    0.008668803609907627,
                    0.2552620768547058,
                    0.9674268960952759,
                    0.9894903302192688
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0113.wav"
            },
            {
                "g_vector": [
                    0.4892788827419281,
                    0.9031171202659607,
                    0.01727639138698578,
                    0.05914677307009697,
                    0.9884774088859558,
                    0.9911671280860901
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0120.wav"
            },
            {
                "g_vector": [
                    0.6820601224899292,
                    0.04886125773191452,
                    0.019520968198776245,
                    0.24887320399284363,
                    0.7435111999511719,
                    0.9802243709564209
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0121.wav"
            },
            {
                "g_vector": [
                    0.9925325512886047,
                    0.9953482747077942,
                    0.276914119720459,
                    0.5435795187950134,
                    0.991669774055481,
                    0.9436368346214294
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0122.wav"
            },
            {
                "g_vector": [
                    0.715868353843689,
                    0.5972997546195984,
                    0.99458247423172,
                    0.3091259300708771,
                    0.8951793313026428,
                    0.9866982102394104
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0123.wav"
            },
            {
                "g_vector": [
                    0.19705210626125336,
                    0.008578155189752579,
                    0.9978992938995361,
                    0.006048895418643951,
                    0.9702776074409485,
                    0.6447311043739319
                ],
                "y_path": "outputs/automated_active_learning/active_learning_inputs/0124.wav"
            }
        ]

"""

# Load the data from the JSON string
data = json.loads(json_data)

# Extract all g_vectors and flatten the list
g_vectors = [item['g_vector'] for item in data]
import numpy as np
all_g_values = np.array(g_vectors).flatten()

plt.rcParams.update({
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# Create the histogram
plt.figure(figsize=(10, 4))
plt.hist(all_g_values, bins=20, edgecolor='black')
# plt.title('Distribution of g-vector Component Values')

# Set the font properties for title and labels
font = {'fontweight': 'bold', 'fontsize': 18}
    

plt.xlabel('Component values of g-vectors', **font)
plt.ylabel('Frequency', **font)
plt.grid(axis='y', alpha=0.75)

# plt.yscale('log')  # Use logarithmic scale for better visibility

plt.savefig('g_vector_distribution.pdf', bbox_inches='tight')