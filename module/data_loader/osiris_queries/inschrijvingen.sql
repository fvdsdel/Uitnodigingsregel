WITH sopl AS (
SELECT 
    sopl.studentnummer
    ,sopl.opleiding
    ,SUBSTR(sopl.opleiding,1,5) AS crebo
    ,sopl.aanvangsjaar_opleiding_student
    ,sopl.aanvangsjaar_opleiding
    ,sopl.cohort_faculteit AS sopl_cohort
    ,sopl.stakingsdatum
    ,sopl.stakingsreden
    ,sopl.instroomsoort
    ,NVL((SELECT 1 FROM ost_student_ook sook 
        WHERE sook.studentnummer = sopl.studentnummer 
        AND sook.opleiding = sopl.opleiding 
        AND sook.actueel_blad_filter = 'J'
        AND sook.ingangsdatum <= sysdate
        AND rownum=1),0) AS actuele_ook
    ,NVL((SELECT 1 FROM ost_student_ook sook 
        WHERE sook.studentnummer = sopl.studentnummer 
        AND sook.opleiding = sopl.opleiding
        AND sook.actueel_blad_filter = 'J'
        AND sook.ingangsdatum > sysdate
        AND rownum=1),0) AS toekomstige_ook
    ,ople.leerweg
    ,ople.niveau
    ,ople.nominale_studieduur
    ,ople.actueel AS ople_actueel
    ,ople.naam_nl
FROM ost_student_opleiding sopl
LEFT JOIN ost_opleiding ople ON (sopl.opleiding = ople.opleiding)
WHERE 1=1
AND NVL(sopl.examendoel,'X') != 'C'
AND SUBSTR(sopl.opleiding,1,1) != 'Z'
AND SUBSTR(sopl.opleiding,1,1) != 'X'
AND ople.leerweg IN ('BOL','BBL')
AND EXISTS(
    SELECT 1 FROM ost_student_ook sook 
    WHERE sook.studentnummer = sopl.studentnummer 
    AND sook.opleiding = sopl.opleiding 
    AND (sook.actueel_blad_filter = 'J' OR sook.instellingstatus = 'BEEINDIGD')
    AND sook.instellingstatus != 'AANGEMELD'
    )
),student_opleiding AS (
SELECT DISTINCT
sopl.studentnummer student
,stud.geslacht
,(FLOOR(MONTHS_BETWEEN(sinh.ingangsdatum,stud.geboortedatum)/12)) AS leeftijd
,sinh.dossier
,sopl.crebo sopl_crebo
,sopl.leerweg
,sopl.niveau
,sopl.sopl_cohort
,sinh.ingangsdatum
,sinh.afloopdatum
,(SELECT organisatieonderdeel FROM ost_student_ook sook WHERE sook.studentnummer = sopl.studentnummer AND sook.opleiding = sopl.opleiding AND rownum=1) AS team
,sopl.actuele_ook
,sopl.toekomstige_ook
,NVL((SELECT 1 FROM ost_student_examen sexa WHERE sexa.studentnummer = sopl.studentnummer AND SUBSTR(sexa.opleiding,1,5)=sopl.crebo AND sexa.examentype='S' AND sexa.judicium='G' AND rownum=1),0) AS diploma
FROM sopl
INNER JOIN (SELECT 
    studentnummer
    ,ople.leerweg
    ,tmbo.externe_code crebo
    ,DECODE(tmbo.element,'KWALIFICATIEDOSSIER',tmbo.externe_code,tmbo2.externe_code) AS dossier
--    ,SUBSTR(sinh.opleiding,1,5) crebo
    ,MIN(sinh.ingangsdatum) ingangsdatum
    ,MAX(sinh.afloopdatum) afloopdatum
FROM ost_student_inschrijfhist sinh
LEFT JOIN ost_opleiding ople ON (sinh.opleiding = ople.opleiding)
LEFT JOIN ost_opleiding_taxonomie otax ON (ople.opleiding = otax.opleiding)
LEFT JOIN ost_taxonomie_mbo tmbo ON (otax.tmbo_id = tmbo.tmbo_id)
LEFT JOIN ost_taxonomie_mbo tmbo2 ON (tmbo.tmbo_id_bovenliggend = tmbo2.tmbo_id)
WHERE 1=1
AND ople.faculteit = 'MBO'
AND SUBSTR(sinh.opleiding,1,1)='2'
GROUP BY studentnummer
,ople.leerweg
,tmbo.externe_code
,DECODE(tmbo.element,'KWALIFICATIEDOSSIER',tmbo.externe_code,tmbo2.externe_code)
HAVING MAX(sinh.afloopdatum) > sysdate-(6*365)
            ) sinh ON (
                    sinh.studentnummer = sopl.studentnummer 
                    AND sinh.crebo = sopl.crebo 
                    AND sinh.leerweg = sopl.leerweg
                    )
LEFT JOIN ost_student stud ON (stud.studentnummer = sopl.studentnummer)
WHERE 1=1
)
SELECT 
    student_opleiding.student studentnummer
    ,geslacht
    ,leeftijd
    ,sopl_crebo
    ,dossier
    ,leerweg
    ,niveau
    ,sopl_cohort
    ,ingangsdatum
    ,EXTRACT(YEAR FROM ingangsdatum) -1 + FLOOR(EXTRACT(MONTH FROM ingangsdatum)/8) AS startjaar
    ,afloopdatum
    ,team
    ,DECODE(geslacht,'M',1,0) AS STUD_GENDER_M
    ,DECODE(geslacht,'V',1,0) AS STUD_GENDER_V
    ,DECODE(geslacht,'O',1,0) AS STUD_GENDER_O
    ,DECODE(student_opleiding.leerweg,'BOL',1,0) AS SOPL_LW_BOL
    ,DECODE(student_opleiding.leerweg,'BBL',1,0) AS SOPL_LW_BBL
    ,DECODE(student_opleiding.niveau,1,1,0) AS SOPL_NIV1
    ,DECODE(student_opleiding.niveau,2,1,0) AS SOPL_NIV2
    ,DECODE(student_opleiding.niveau,3,1,0) AS SOPL_NIV3
    ,DECODE(student_opleiding.niveau,4,1,0) AS SOPL_NIV4
    ,actuele_ook
    ,toekomstige_ook
    ,diploma
    ,student_opleiding.actuele_ook+student_opleiding.toekomstige_ook+diploma AS geen_uitval
FROM student_opleiding
WHERE 1=1
AND sopl_cohort >= 2020
AND ingangsdatum > sysdate - (6*365)
AND afloopdatum-ingangsdatum > 20
AND sopl_cohort <= EXTRACT(YEAR FROM ingangsdatum)