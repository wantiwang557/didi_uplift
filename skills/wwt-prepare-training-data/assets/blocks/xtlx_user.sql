LEFT JOIN (
 SELECT * FROM dm.xtlx_user_test_v1
 WHERE dt BETWEEN '{{source_start}}' AND '{{source_end}}'
) ttt ON t.pid = ttt.pid AND t.dt = date_add(ttt.dt, 1)
