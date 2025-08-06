WITH dummy AS (SELECT 1 FROM DUAL)

,schooljaar AS (
SELECT 2022 AS jaar, 35 AS startweek,42 AS herfst FROM DUAL
UNION SELECT 2023 AS jaar, 36 AS startweek ,43 AS herfst FROM DUAL
UNION SELECT 2024 AS jaar, 36 AS startweek ,44 AS herfst FROM DUAL
)

,verzuim AS (
SELECT 
saaw.studentnummer
,saaw.datum
,EXTRACT(YEAR FROM saaw.datum) AS kalenderjaar
,EXTRACT(MONTH FROM saaw.datum) AS maand
,EXTRACT(YEAR FROM saaw.datum)-1+FLOOR(EXTRACT(MONTH FROM saaw.datum)/8) AS schooljaar
,TO_NUMBER(TO_CHAR(saaw.datum,'IW')) AS weeknummer
,saaw.minuten_aanwezig
,saaw.minuten_geoorloofd
,saaw.minuten_ongeoorloofd
--,(saaw.minuten_aanwezig+saaw.minuten_geoorloofd+saaw.minuten_ongeoorloofd) AS minuten_reg
FROM ost_student_aanwezigheid saaw
)
,stud_verzuim AS (
SELECT 
studentnummer
,schooljaar
-- ,(weeknummer-sj.startweek+1) AS schoolweek
--,weeknummer
,CASE WHEN weeknummer < sj.herfst THEN (weeknummer-sj.startweek+1) 
    WHEN weeknummer > sj.herfst THEN (weeknummer-sj.startweek) 
    ELSE NULL
    END AS schoolweek

,NVL(minuten_aanwezig,0) AS totaal_aanw
,NVL(minuten_aanwezig,0)+NVL(minuten_geoorloofd,0)+NVL(minuten_ongeoorloofd,0) AS totaal_reg
,NVL(minuten_geoorloofd,0) AS totaal_verzuim_ap
,NVL(minuten_ongeoorloofd,0) AS totaal_verzuim_unap
FROM verzuim LEFT JOIN schooljaar sj ON (sj.jaar = verzuim.schooljaar)
WHERE verzuim.kalenderjaar = verzuim.schooljaar
-- AND (weeknummer-sj.startweek+1) < 11
AND NVL(CASE WHEN weeknummer < sj.herfst THEN (weeknummer-sj.startweek+1) 
    WHEN weeknummer > sj.herfst THEN (weeknummer-sj.startweek) 
    ELSE NULL
    END,12) < 11
)
,vz AS (
    SELECT stud_verzuim.*
FROM stud_verzuim
)

, vz_agg AS (
SELECT 
studentnummer,schooljaar,schoolweek
,SUM(totaal_aanw) AS  totaal_aanw
,SUM(totaal_reg) AS  totaal_reg
,SUM(totaal_verzuim_ap) AS  totaal_verzuim_ap
,SUM(totaal_verzuim_unap) AS  totaal_verzuim_unap
FROM stud_verzuim
GROUP BY studentnummer,schooljaar,schoolweek)

,vz_agg2 AS (
SELECT * FROM vz_agg
PIVOT(
SUM(NVL(totaal_aanw,0)) AS Aanwezig
,SUM(totaal_reg) AS TotaalReg
,SUM(totaal_verzuim_ap) AS Approved
,SUM(totaal_verzuim_unap) AS Unapproved
,SUM(ROUND(totaal_aanw/totaal_reg,3)) AS TotaalAanwPct
,SUM(ROUND(totaal_verzuim_ap/totaal_reg,3)) AS VerzuimAp
,SUM(ROUND(totaal_verzuim_unap/totaal_reg,3)) AS VerzuimUnap

FOR schoolweek IN (
    1   AS VERZ_W1
    ,2   AS VERZ_W2
    ,3   AS VERZ_W3
    ,4   AS VERZ_W4
    ,5   AS VERZ_W5
    ,6   AS VERZ_W6
    ,7   AS VERZ_W7
    ,8   AS VERZ_W8
    ,9   AS VERZ_W9
    ,10   AS VERZ_W10
    )
)
)
, agg_totaal AS (
SELECT 
vz_agg2.* 
,NVL(vz_agg2.VERZ_W1_AANWEZIG,0) + NVL(vz_agg2.VERZ_W2_AANWEZIG,0) + NVL(vz_agg2.VERZ_W3_AANWEZIG,0)
    -- + VERZ_W4_Aanwezig + VERZ_W5_Aanwezig + VERZ_W6_Aanwezig + VERZ_W7_Aanwezig + VERZ_W8_Aanwezig + VERZ_W9_Aanwezig + VERZ_W10_Aanwezig
     AS TotaalAanw3
,NVL(vz_agg2.VERZ_W1_AANWEZIG,0) + NVL(vz_agg2.VERZ_W2_AANWEZIG,0) + NVL(vz_agg2.VERZ_W3_AANWEZIG,0)
    + NVL(VERZ_W4_AANWEZIG,0) + NVL(VERZ_W5_AANWEZIG,0) + NVL(VERZ_W6_AANWEZIG,0)
    --  + VERZ_W7_Aanwezig + VERZ_W8_Aanwezig + VERZ_W9_Aanwezig + VERZ_W10_Aanwezig
     AS TotaalAanw6
,NVL(vz_agg2.VERZ_W1_AANWEZIG,0) + NVL(vz_agg2.VERZ_W2_AANWEZIG,0) + NVL(vz_agg2.VERZ_W3_AANWEZIG,0)
    + NVL(VERZ_W4_AANWEZIG,0) + NVL(VERZ_W5_AANWEZIG,0) + NVL(VERZ_W6_AANWEZIG,0)
     + NVL(VERZ_W7_AANWEZIG,0) + NVL(VERZ_W8_AANWEZIG,0) + NVL(VERZ_W9_AANWEZIG,0) + NVL(VERZ_W10_AANWEZIG,0)
     AS TotaalAanw10

--
,NVL(vz_agg2.VERZ_W1_TOTAALREG,0) + NVL(vz_agg2.VERZ_W2_TOTAALREG,0) + NVL(vz_agg2.VERZ_W3_TOTAALREG,0)
    -- + VERZ_W4_Aanwezig + VERZ_W5_Aanwezig + VERZ_W6_Aanwezig + VERZ_W7_Aanwezig + VERZ_W8_Aanwezig + VERZ_W9_Aanwezig + VERZ_W10_Aanwezig
     AS TotaalReg3
,NVL(vz_agg2.VERZ_W1_TOTAALREG,0) + NVL(vz_agg2.VERZ_W2_TOTAALREG,0) + NVL(vz_agg2.VERZ_W3_TOTAALREG,0)
    + NVL(VERZ_W4_TOTAALREG,0) + NVL(VERZ_W5_TOTAALREG,0) + NVL(VERZ_W6_TOTAALREG,0)
    --  + VERZ_W7_Aanwezig + VERZ_W8_Aanwezig + VERZ_W9_Aanwezig + VERZ_W10_Aanwezig
     AS TotaalReg6
,NVL(vz_agg2.VERZ_W1_TOTAALREG,0) + NVL(vz_agg2.VERZ_W2_TOTAALREG,0) + NVL(vz_agg2.VERZ_W3_TOTAALREG,0)
    + NVL(VERZ_W4_TOTAALREG,0) + NVL(VERZ_W5_TOTAALREG,0) + NVL(VERZ_W6_TOTAALREG,0)
     + NVL(VERZ_W7_TOTAALREG,0) + NVL(VERZ_W8_TOTAALREG,0) + NVL(VERZ_W9_TOTAALREG,0) + NVL(VERZ_W10_TOTAALREG,0)
     AS TotaalReg10
FROM vz_agg2
)

SELECT 
agg_totaal.* 
,ROUND(totaalaanw3/DECODE(totaalreg3,0,1,totaalreg3),2) AS aanw_pct3
,ROUND(totaalaanw6/DECODE(totaalreg6,0,1,totaalreg6),2) AS aanw_pct6
,ROUND(totaalaanw10/DECODE(totaalreg10,0,1,totaalreg10),2) AS aanw_pct10
FROM agg_totaal
WHERE 1=1
-- WHERE schooljaar = 2023