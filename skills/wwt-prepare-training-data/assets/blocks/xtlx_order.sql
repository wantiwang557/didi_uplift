LEFT JOIN (
 SELECT * FROM dm.xtlx_wchs_test_v1
 WHERE dt BETWEEN '{{source_start}}' AND '{{source_end}}'
) tt ON t.pid = tt.passenger_id AND t.dt = date_add(tt.dt, 1)
