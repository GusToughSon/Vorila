#include <iostream>

#include "main.h"


/* Info about what will be in MonsterInfo array

	Name    
	Avatar
	DamMin
	DamMax
	HPMax
	AC
	Dex
	ExpMin
	ExpMax
	MovingSpeed
	AttackSpeed
	MonsterLevel
	ElementalDefence
	DefenceAmount
	ElementalAttack
	AttackAmount
	Loot[10]
*/


unsigned long ExpTables[MaxLevel];
unsigned long SkillExpTables[MaxLevel];
unsigned long AligmentTables[12];

void CreateTables()
{
	ExpTables[0] = 0;
	ExpTables[1] = 0;
	ExpTables[2] = 300;
	ExpTables[3] = 600;
	ExpTables[4] = 1000;
	ExpTables[5] = 1500;
	ExpTables[6] = 2000;
	ExpTables[7] = 4000;
	ExpTables[8] = 6000;
	ExpTables[9] = 8000;
	ExpTables[10] = 10000;
	ExpTables[11] = 20000;
	ExpTables[12] = 30000;
	ExpTables[13] = 40000;
	ExpTables[14] = 50000;
	ExpTables[15] = 60000;
	ExpTables[16] = 75000;
	ExpTables[17] = 100000;
	ExpTables[18] = 125000;
	ExpTables[19] = 150000;
	ExpTables[20] = 200000;
	ExpTables[21] = 250000;
	ExpTables[22] = 300000;
	ExpTables[23] = 400000;
	ExpTables[24] = 500000;
	ExpTables[25] = 600000;
	ExpTables[26] = 800000;
	ExpTables[27] = 1000000;
	ExpTables[28] = 1200000;
	ExpTables[29] = 1600000;
	ExpTables[30] = 2000000;
	ExpTables[31] = 2400000;
	ExpTables[32] = 2800000;
	ExpTables[33] = 3200000;
	ExpTables[34] = 3600000;
	ExpTables[35] = 4000000;
	ExpTables[36] = 4400000;
	ExpTables[37] = 4800000;
	ExpTables[38] = 5200000;
	ExpTables[39] = 5600000;
	ExpTables[40] = 6000000;
	ExpTables[41] = 6400000;
	ExpTables[42] = 6800000;
	ExpTables[43] = 7200000;
	ExpTables[44] = 7600000;
	ExpTables[45] = 8000000;
	ExpTables[46] = 8400000;
	ExpTables[47] = 8800000;
	ExpTables[48] = 9200000;
	ExpTables[49] = 9600000;
	ExpTables[50] = 10000000;
	ExpTables[51] = 11000000;
	ExpTables[52] = 12000000;
	ExpTables[53] = 13000000;
	ExpTables[54] = 14000000;
	ExpTables[55] = 15000000;
	ExpTables[56] = 16000000;
	ExpTables[57] = 17000000;
	ExpTables[58] = 18000000;
	ExpTables[59] = 19000000;
	ExpTables[60] = 20000000;
	ExpTables[61] = 22000000;
	ExpTables[62] = 24000000;
	ExpTables[63] = 26000000;
	ExpTables[64] = 28000000;
	ExpTables[65] = 30000000;
	ExpTables[66] = 32000000;
	ExpTables[67] = 34000000;
	ExpTables[68] = 36000000;
	ExpTables[69] = 38000000;
	ExpTables[70] = 40000000;
	ExpTables[71] = 43000000;
	ExpTables[72] = 46000000;
	ExpTables[73] = 49000000;
	ExpTables[74] = 52000000;
	ExpTables[75] = 55000000;
	ExpTables[76] = 58000000;
	ExpTables[77] = 61000000;
	ExpTables[78] = 64000000;
	ExpTables[79] = 67000000;
	ExpTables[80] = 70000000;
	ExpTables[81] = 74000000;
	ExpTables[82] = 78000000;
	ExpTables[83] = 82000000;
	ExpTables[84] = 86000000;
	ExpTables[85] = 90000000;
	ExpTables[86] = 94000000;
	ExpTables[87] = 98000000;
	ExpTables[88] = 102000000;
	ExpTables[89] = 106000000;
	ExpTables[90] = 110000000;
	ExpTables[91] = 115000000;
	ExpTables[92] = 120000000;
	ExpTables[93] = 125000000;
	ExpTables[94] = 130000000;
	ExpTables[95] = 135000000;
	ExpTables[96] = 140000000;
	ExpTables[97] = 145000000;
	ExpTables[98] = 150000000;
	ExpTables[99] = 155000000;
	ExpTables[100] = 160000000;
	ExpTables[101] = 170000000;
	ExpTables[102] = 180000000;
	ExpTables[103] = 190000000;
	ExpTables[104] = 200000000;
	ExpTables[105] = 210000000;
	ExpTables[106] = 220000000;
	ExpTables[107] = 230000000;
	ExpTables[108] = 240000000;
	ExpTables[109] = 250000000;
	ExpTables[110] = 260000000;
	ExpTables[111] = 270000000;
	ExpTables[112] = 280000000;
	ExpTables[113] = 290000000;
	ExpTables[114] = 300000000;
	ExpTables[115] = 310000000;
	ExpTables[116] = 320000000;
	ExpTables[117] = 330000000;
	ExpTables[118] = 340000000;
	ExpTables[119] = 350000000;
	ExpTables[120] = 360000000;
	ExpTables[121] = 370000000;
	ExpTables[122] = 380000000;
	ExpTables[123] = 390000000;
	ExpTables[124] = 400000000;
	ExpTables[125] = 410000000;
	ExpTables[126] = 420000000;
	ExpTables[127] = 430000000;
	ExpTables[128] = 440000000;
	ExpTables[129] = 450000000;
	ExpTables[130] = 460000000;
	ExpTables[131] = 470000000;
	ExpTables[132] = 480000000;
	ExpTables[133] = 490000000;
	ExpTables[134] = 500000000;
	ExpTables[135] = 510000000;
	ExpTables[136] = 520000000;
	ExpTables[137] = 530000000;
	ExpTables[138] = 540000000;
	ExpTables[139] = 550000000;
	ExpTables[140] = 560000000;
	ExpTables[141] = 570000000;
	ExpTables[142] = 580000000;
	ExpTables[143] = 590000000;
	ExpTables[144] = 600000000;
	ExpTables[145] = 620000000;
	ExpTables[146] = 640000000;
	ExpTables[147] = 660000000;
	ExpTables[148] = 680000000;
	ExpTables[149] = 700000000;
	ExpTables[150] = 720000000;
	ExpTables[151] = 740000000;
	ExpTables[152] = 760000000;
	ExpTables[153] = 780000000;
	ExpTables[154] = 800000000;
	ExpTables[155] = 820000000;
	ExpTables[156] = 840000000;
	ExpTables[157] = 860000000;
	ExpTables[158] = 880000000;
	ExpTables[159] = 900000000;
	ExpTables[160] = 920000000;
	ExpTables[161] = 940000000;
	ExpTables[162] = 960000000;
	ExpTables[163] = 980000000;
	ExpTables[164] = 1000000000;
	ExpTables[165] = 1020000000;
	ExpTables[166] = 1040000000;
	ExpTables[167] = 1060000000;
	ExpTables[168] = 1080000000;
	ExpTables[169] = 1100000000;
	ExpTables[170] = 1120000000;
	ExpTables[171] = 1140000000;
	ExpTables[172] = 1160000000;
	ExpTables[173] = 1180000000;
	ExpTables[174] = 1200000000;
	ExpTables[175] = 1220000000;
	ExpTables[176] = 1240000000;
	ExpTables[177] = 1260000000;
	ExpTables[178] = 1280000000;
	ExpTables[179] = 1300000000;
	ExpTables[180] = 1320000000;
	ExpTables[181] = 1340000000;
	ExpTables[182] = 1360000000;
	ExpTables[183] = 1380000000;
	ExpTables[184] = 1400000000;
	ExpTables[185] = 1420000000;
	ExpTables[186] = 1440000000;
	ExpTables[187] = 1460000000;
	ExpTables[188] = 1480000000;
	ExpTables[189] = 1500000000;
	ExpTables[190] = 1520000000;
	ExpTables[191] = 1540000000;
	ExpTables[192] = 1560000000;
	ExpTables[193] = 1580000000;
	ExpTables[194] = 1600000000;
	ExpTables[195] = 1620000000;
	ExpTables[196] = 1640000000;
	ExpTables[197] = 1660000000;
	ExpTables[198] = 1680000000;
	ExpTables[199] = 1700000000;
	ExpTables[200] = 1750000000;
	ExpTables[201] = 4200000000;


	SkillExpTables[0] = 0;
	SkillExpTables[1] = 150;
	SkillExpTables[2] = 300;
	SkillExpTables[3] = 600;
	SkillExpTables[4] = 1000;
	SkillExpTables[5] = 1500;
	SkillExpTables[6] = 2500;
	SkillExpTables[7] = 4000;
	SkillExpTables[8] = 6000;
	SkillExpTables[9] = 8000;
	SkillExpTables[10] = 10000;
	SkillExpTables[11] = 14000;
	SkillExpTables[12] = 18000;
	SkillExpTables[13] = 24000;
	SkillExpTables[14] = 30000;
	SkillExpTables[15] = 40000;
	SkillExpTables[16] = 50000;
	SkillExpTables[17] = 60000;
	SkillExpTables[18] = 70000;
	SkillExpTables[19] = 80000;
	SkillExpTables[20] = 90000;
	SkillExpTables[21] = 100000;
	SkillExpTables[22] = 125000;
	SkillExpTables[23] = 150000;
	SkillExpTables[24] = 175000;
	SkillExpTables[25] = 200000;
	SkillExpTables[26] = 250000;
	SkillExpTables[27] = 300000;
	SkillExpTables[28] = 350000;
	SkillExpTables[29] = 400000;
	SkillExpTables[30] = 450000;
	SkillExpTables[31] = 500000;
	SkillExpTables[32] = 550000;
	SkillExpTables[33] = 600000;
	SkillExpTables[34] = 650000;
	SkillExpTables[35] = 700000;
	SkillExpTables[36] = 750000;
	SkillExpTables[37] = 800000;
	SkillExpTables[38] = 850000;
	SkillExpTables[39] = 900000;
	SkillExpTables[40] = 975000;
	SkillExpTables[41] = 1050000;
	SkillExpTables[42] = 1125000;
	SkillExpTables[43] = 1200000;
	SkillExpTables[44] = 1300000;
	SkillExpTables[45] = 1400000;
	SkillExpTables[46] = 1500000;
	SkillExpTables[47] = 1600000;
	SkillExpTables[48] = 1700000;
	SkillExpTables[49] = 1800000;
	SkillExpTables[50] = 1900000;
	SkillExpTables[51] = 2000000;
	SkillExpTables[52] = 2100000;
	SkillExpTables[53] = 2200000;
	SkillExpTables[54] = 2300000;
	SkillExpTables[55] = 2400000;
	SkillExpTables[56] = 2500000;
	SkillExpTables[57] = 2600000;
	SkillExpTables[58] = 2700000;
	SkillExpTables[59] = 2800000;
	SkillExpTables[60] = 2900000;
	SkillExpTables[61] = 3000000;
	SkillExpTables[62] = 3250000;
	SkillExpTables[63] = 3500000;
	SkillExpTables[64] = 3750000;
	SkillExpTables[65] = 4000000;
	SkillExpTables[66] = 4250000;
	SkillExpTables[67] = 4500000;
	SkillExpTables[68] = 4750000;
	SkillExpTables[69] = 5000000;
	SkillExpTables[70] = 5250000;
	SkillExpTables[71] = 5500000;
	SkillExpTables[72] = 6000000;
	SkillExpTables[73] = 6500000;
	SkillExpTables[74] = 7000000;
	SkillExpTables[75] = 7500000;
	SkillExpTables[76] = 8000000;
	SkillExpTables[77] = 8500000;
	SkillExpTables[78] = 9000000;
	SkillExpTables[79] = 9500000;
	SkillExpTables[80] = 10000000;
	SkillExpTables[81] = 11000000;
	SkillExpTables[82] = 12000000;
	SkillExpTables[83] = 13000000;
	SkillExpTables[84] = 14000000;
	SkillExpTables[85] = 15000000;
	SkillExpTables[86] = 16000000;
	SkillExpTables[87] = 17000000;
	SkillExpTables[88] = 18000000;
	SkillExpTables[89] = 19000000;
	SkillExpTables[90] = 20000000;
	SkillExpTables[91] = 22000000;
	SkillExpTables[92] = 24000000;
	SkillExpTables[93] = 26000000;
	SkillExpTables[94] = 28000000;
	SkillExpTables[95] = 30000000;
	SkillExpTables[96] = 35000000;
	SkillExpTables[97] = 40000000;
	SkillExpTables[98] = 50000000;
	SkillExpTables[99] = 60000000;
	SkillExpTables[100] = 80000000;	
	SkillExpTables[101] = 1000000000;
	SkillExpTables[102] = 2000000000;
	SkillExpTables[103] = 3000000000;


	AligmentTables[0] = 0; // scourge
	AligmentTables[1] = 500; // evil
	AligmentTables[2] = 1500; // hated
	AligmentTables[3] = 3000; // disliked
	AligmentTables[4] = 3500; // neutral
	AligmentTables[5] = 4000; // liked
	AligmentTables[6] = 5500; // good
	AligmentTables[7] = 7000; // hero
	AligmentTables[8] = 20000; // divine
	AligmentTables[9] = 1000000000; // nothing
	AligmentTables[10] = 1000000000; // normal
	AligmentTables[11] = 1000000000; // GMIT
}


