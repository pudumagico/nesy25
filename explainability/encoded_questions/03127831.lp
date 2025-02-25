% ------ scene encoding ------
object(405481).
has_attr(405481, class, head).
has_attr(405481, name, head).
has_attr(405481, hposition, left).
has_attr(405481, vposition, middle).
object(405471).
has_attr(405471, class, sheep).
has_attr(405471, name, sheep).
has_attr(405471, class, animal).
has_attr(405471, pose, standing).
has_attr(405471, any, shaggy).
has_attr(405471, pose, sitting).
has_attr(405471, color, beige).
has_attr(405471, any, grazing).
has_attr(405471, hposition, left).
has_attr(405471, vposition, top).
has_rel(405471, in, 405470).
has_rel(405471, to_the_left_of, 405472).
object(405477).
has_attr(405477, class, rock).
has_attr(405477, name, rock).
has_attr(405477, class, thing).
has_attr(405477, class, material).
has_attr(405477, color, brown).
has_attr(405477, hposition, left).
has_attr(405477, vposition, bottom).
has_attr(405477, vposition, middle).
object(405472).
has_attr(405472, class, goat).
has_attr(405472, name, goat).
has_attr(405472, class, animal).
has_attr(405472, color, brown).
has_attr(405472, activity, eating).
has_attr(405472, hposition, middle).
has_attr(405472, vposition, top).
has_rel(405472, standing_in, 405470).
has_rel(405472, in, 405470).
has_rel(405472, eating, 405470).
has_rel(405472, to_the_right_of, 405479).
has_rel(405472, to_the_right_of, 405471).
object(405470).
has_attr(405470, class, grass).
has_attr(405470, name, grass).
has_attr(405470, class, plant).
has_attr(405470, class, place).
has_attr(405470, height, tall).
has_attr(405470, color, green).
has_attr(405470, hposition, middle).
has_attr(405470, vposition, bottom).
has_attr(405470, vposition, middle).
object(405480).
has_attr(405480, class, nose).
has_attr(405480, name, nose).
has_attr(405480, color, black).
has_attr(405480, hposition, left).
has_attr(405480, vposition, bottom).
has_attr(405480, vposition, middle).
object(405483).
has_attr(405483, class, ear).
has_attr(405483, name, ear).
has_attr(405483, hposition, left).
has_attr(405483, vposition, middle).
has_rel(405483, to_the_right_of, 405482).
object(405482).
has_attr(405482, class, eye).
has_attr(405482, name, eye).
has_attr(405482, hposition, left).
has_attr(405482, vposition, middle).
has_rel(405482, to_the_left_of, 405483).
object(405485).
has_attr(405485, class, leg).
has_attr(405485, name, leg).
has_attr(405485, hposition, left).
has_attr(405485, vposition, top).
object(405484).
has_attr(405484, class, mouth).
has_attr(405484, name, mouth).
has_attr(405484, hposition, left).
has_attr(405484, vposition, bottom).
has_attr(405484, vposition, middle).
object(405486).
has_attr(405486, class, leaf).
has_attr(405486, name, leaf).
has_attr(405486, class, thing).
has_attr(405486, color, green).
has_attr(405486, hposition, right).
has_attr(405486, vposition, middle).
object(405478).
has_attr(405478, class, dirt).
has_attr(405478, name, dirt).
has_attr(405478, class, place).
has_attr(405478, class, thing).
has_attr(405478, color, brown).
has_attr(405478, any, patched).
has_attr(405478, hposition, middle).
has_attr(405478, vposition, bottom).
has_attr(405478, vposition, middle).
object(405479).
has_attr(405479, class, tail).
has_attr(405479, name, tail).
has_attr(405479, height, short).
has_attr(405479, length, short).
has_attr(405479, hposition, left).
has_attr(405479, vposition, top).
has_rel(405479, to_the_left_of, 405472).
object(405473).
has_attr(405473, class, sheep).
has_attr(405473, name, sheep).
has_attr(405473, class, animal).
has_attr(405473, pose, lying).
has_attr(405473, hposition, middle).
has_attr(405473, vposition, middle).
has_rel(405473, to_the_right_of, 405475).
object(405475).
has_attr(405475, class, sheep).
has_attr(405475, name, sheep).
has_attr(405475, class, animal).
has_attr(405475, hposition, left).
has_attr(405475, vposition, middle).
has_rel(405475, to_the_left_of, 405473).

% ------ question encoding ------
scene(0).
select(1, 0, grass).
relate(2, 1, animal, eating, subject).
unique(3, 2).
query(4, 3, name).
end(4).
% ------ NL question ------

% What is the animal that is eating the grass?

% ------ Full answer ------

% The animal is a goat.
