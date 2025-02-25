% ------ scene encoding ------
object(557782).
has_attr(557782, class, girl).
has_attr(557782, name, girl).
has_attr(557782, class, person).
has_attr(557782, class, she).
has_attr(557782, size, little).
has_attr(557782, age, little).
has_attr(557782, color, blond).
has_attr(557782, hposition, left).
has_attr(557782, vposition, bottom).
has_attr(557782, vposition, middle).
has_rel(557782, in, 557783).
has_rel(557782, to_the_left_of, 557798).
has_rel(557782, to_the_right_of, 557789).
has_rel(557782, to_the_right_of, 557791).
object(557783).
has_attr(557783, class, grass).
has_attr(557783, name, grass).
has_attr(557783, class, plant).
has_attr(557783, class, place).
has_attr(557783, color, brown).
has_attr(557783, hposition, middle).
has_attr(557783, vposition, bottom).
has_attr(557783, vposition, middle).
has_rel(557783, on, 3836031).
object(557793).
has_attr(557793, class, tail).
has_attr(557793, name, tail).
has_attr(557793, hposition, middle).
has_attr(557793, vposition, top).
object(557795).
has_attr(557795, class, jacket).
has_attr(557795, name, jacket).
has_attr(557795, class, clothing).
has_attr(557795, color, black).
has_attr(557795, hposition, middle).
has_attr(557795, vposition, bottom).
has_attr(557795, vposition, middle).
has_rel(557795, to_the_left_of, 557787).
has_rel(557795, to_the_right_of, 557788).
object(557787).
has_attr(557787, class, jacket).
has_attr(557787, name, jacket).
has_attr(557787, class, clothing).
has_attr(557787, color, brown).
has_attr(557787, hposition, right).
has_attr(557787, vposition, bottom).
has_attr(557787, vposition, middle).
has_rel(557787, to_the_right_of, 557795).
object(557784).
has_attr(557784, class, cloud).
has_attr(557784, name, cloud).
has_attr(557784, class, thing).
has_attr(557784, color, dark).
has_attr(557784, tone, dark).
has_attr(557784, hposition, middle).
has_attr(557784, vposition, top).
object(557790).
has_attr(557790, class, sky).
has_attr(557790, name, sky).
has_attr(557790, class, nature_environment).
has_attr(557790, weather, overcast).
has_attr(557790, weather, cloudy).
has_attr(557790, hposition, middle).
has_attr(557790, vposition, middle).
object(3836031).
has_attr(3836031, class, ground).
has_attr(3836031, name, ground).
has_attr(3836031, class, place).
has_attr(3836031, class, nature_environment).
has_attr(3836031, hposition, middle).
has_attr(3836031, vposition, bottom).
has_attr(3836031, vposition, middle).
object(557788).
has_attr(557788, class, tree).
has_attr(557788, name, tree).
has_attr(557788, class, tree).
has_attr(557788, hposition, middle).
has_attr(557788, vposition, bottom).
has_attr(557788, vposition, middle).
has_rel(557788, to_the_right_of, 557791).
has_rel(557788, to_the_left_of, 557795).
has_rel(557788, to_the_right_of, 557789).
object(557789).
has_attr(557789, class, tent).
has_attr(557789, name, tent).
has_attr(557789, class, thing).
has_attr(557789, color, blue).
has_attr(557789, hposition, left).
has_attr(557789, vposition, bottom).
has_attr(557789, vposition, middle).
has_rel(557789, to_the_left_of, 557791).
has_rel(557789, to_the_left_of, 557788).
has_rel(557789, to_the_left_of, 557782).
object(557798).
has_attr(557798, class, child).
has_attr(557798, name, child).
has_attr(557798, class, person).
has_attr(557798, hposition, middle).
has_attr(557798, vposition, bottom).
has_attr(557798, vposition, middle).
has_rel(557798, sitting_on, 557783).
has_rel(557798, to_the_right_of, 557782).
has_rel(557798, to_the_right_of, 557791).
object(557786).
has_attr(557786, class, kite).
has_attr(557786, name, kite).
has_attr(557786, class, thing).
has_attr(557786, any, colorful).
has_attr(557786, any, rainbow_colored).
has_attr(557786, hposition, left).
has_attr(557786, vposition, middle).
object(557785).
has_attr(557785, class, kite).
has_attr(557785, name, kite).
has_attr(557785, class, thing).
has_attr(557785, color, red).
has_attr(557785, hposition, middle).
has_attr(557785, vposition, top).
has_rel(557785, in, 557790).
has_rel(557785, flying_in, 557790).
object(557791).
has_attr(557791, class, person).
has_attr(557791, name, person).
has_attr(557791, class, person).
has_attr(557791, pose, standing).
has_attr(557791, hposition, left).
has_attr(557791, vposition, bottom).
has_attr(557791, vposition, middle).
has_rel(557791, flying, 557786).
has_rel(557791, holding, 557786).
has_rel(557791, walking_on, 557783).
has_rel(557791, to_the_left_of, 557788).
has_rel(557791, to_the_left_of, 557798).
has_rel(557791, to_the_left_of, 557782).
has_rel(557791, to_the_right_of, 557789).

% ------ question encoding ------
scene(0).
select(1, 0, tent).
unique(2, 1).
choose_attr(3, 2, vposition, bottom, top).
end(3).
% ------ NL question ------

% In which part of the picture is the tent, the bottom or the top?

% ------ Full answer ------

% The tent is in the bottom of the image.
