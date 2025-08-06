WITH dossiers AS (
SELECT 
tmbo.externe_code   AS opleiding_crebo
,DECODE(tmbo.element,'KWALIFICATIEDOSSIER',1,0) AS dossier
,DECODE(tmbo.element,'KWALIFICATIE',1,0) AS kwalificatie
,DECODE(tmbo.element,'KWALIFICATIEDOSSIER',tmbo.externe_code,tmbo2.externe_code) AS dossier_crebo
FROM ost_taxonomie_mbo tmbo
LEFT JOIN ost_taxonomie_mbo tmbo2 ON (tmbo.tmbo_id_bovenliggend = tmbo2.tmbo_id AND tmbo.element = 'KWALIFICATIE')
WHERE tmbo.element IN ('KWALIFICATIEDOSSIER','KWALIFICATIE')
)
SELECT dossiers.* 
,tmbo.diplomanaam
,DECODE(tmbo2.diplomanaam,NULL,tmbo2.naam,'OVERIG') AS domein
FROM ost_taxonomie_mbo tmbo
INNER JOIN dossiers ON (tmbo.externe_code = dossiers.dossier_crebo)
LEFT JOIN ost_taxonomie_mbo tmbo2 ON (tmbo.tmbo_id_bovenliggend = tmbo2.tmbo_id)