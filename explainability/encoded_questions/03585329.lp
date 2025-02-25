% ------ scene encoding ------
object(936994).
has_attr(936994, class, surfboard).
has_attr(936994, name, surfboard).
has_attr(936994, class, object).
has_attr(936994, color, blue).
has_attr(936994, hposition, middle).
has_attr(936994, vposition, middle).
has_rel(936994, to_the_right_of, 937000).
object(936986).
has_attr(936986, class, water).
has_attr(936986, name, water).
has_attr(936986, class, liquid).
has_attr(936986, any, foamy).
has_attr(936986, hposition, right).
has_attr(936986, vposition, middle).
has_rel(936986, to_the_right_of, 936993).
has_rel(936986, to_the_right_of, 936991).
has_rel(936986, to_the_right_of, 937003).
object(936993).
has_attr(936993, class, girl).
has_attr(936993, name, girl).
has_attr(936993, class, person).
has_attr(936993, class, she).
has_attr(936993, hposition, left).
has_attr(936993, vposition, middle).
has_rel(936993, on, 936994).
has_rel(936993, to_the_left_of, 936986).
has_rel(936993, to_the_right_of, 937000).
object(936990).
has_attr(936990, class, wetsuit).
has_attr(936990, name, wetsuit).
has_attr(936990, class, clothing).
has_attr(936990, hposition, middle).
has_attr(936990, vposition, bottom).
has_attr(936990, vposition, middle).
object(936991).
has_attr(936991, class, person).
has_attr(936991, name, person).
has_attr(936991, class, person).
has_attr(936991, hposition, middle).
has_attr(936991, vposition, top).
has_rel(936991, to_the_right_of, 937000).
has_rel(936991, standing_in, 937000).
has_rel(936991, to_the_left_of, 936986).
object(936989).
has_attr(936989, class, boy).
has_attr(936989, name, boy).
has_attr(936989, class, person).
has_attr(936989, class, he).
has_attr(936989, any, smiling).
has_attr(936989, pose, lying).
has_attr(936989, any, happy).
has_attr(936989, hposition, middle).
has_attr(936989, vposition, middle).
object(937003).
has_attr(937003, class, shoe).
has_attr(937003, name, shoe).
has_attr(937003, class, footwear).
has_attr(937003, hposition, middle).
has_attr(937003, vposition, middle).
has_rel(937003, to_the_left_of, 936986).
has_rel(937003, to_the_right_of, 937000).
object(937001).
has_attr(937001, class, child).
has_attr(937001, name, child).
has_attr(937001, class, person).
has_attr(937001, any, floating).
has_attr(937001, pose, lying).
has_attr(937001, sportactivity, swimming).
has_attr(937001, sportactivity, swimming).
has_attr(937001, hposition, middle).
has_attr(937001, vposition, middle).
has_rel(937001, wearing, 937003).
has_rel(937001, in, 937000).
has_rel(937001, to_the_right_of, 937000).
has_rel(937001, in, 936986).
object(937000).
has_attr(937000, class, ocean).
has_attr(937000, name, ocean).
has_attr(937000, class, place).
has_attr(937000, hposition, left).
has_attr(937000, vposition, middle).
has_rel(937000, to_the_left_of, 936991).
has_rel(937000, to_the_left_of, 936993).
has_rel(937000, to_the_left_of, 937003).
has_rel(937000, to_the_left_of, 937001).
has_rel(937000, to_the_left_of, 936994).

% ------ question encoding ------
scene(0).
select(1, 0, girl).
relate_any(2, 1, on, object).
unique(3, 2).
query(4, 3, name).
end(4).
% ------ NL question ------

% What is the girl on?

% ------ Full answer ------

% The girl is on the surfboard.
