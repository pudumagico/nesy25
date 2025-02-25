% ------ scene encoding ------
object(1243751).
has_attr(1243751, class, bread).
has_attr(1243751, name, bread).
has_attr(1243751, class, baked_good).
has_attr(1243751, tone, light).
has_attr(1243751, weight, light).
has_attr(1243751, any, crispy).
has_attr(1243751, any, toasted).
has_attr(1243751, hposition, middle).
has_attr(1243751, vposition, middle).
has_rel(1243751, above, 1243746).
object(1243750).
has_attr(1243750, class, finger).
has_attr(1243750, name, finger).
has_attr(1243750, color, white).
has_attr(1243750, hposition, left).
has_attr(1243750, vposition, top).
has_rel(1243750, touching, 1243745).
object(1243746).
has_attr(1243746, class, egg).
has_attr(1243746, name, egg).
has_attr(1243746, class, food).
has_attr(1243746, any, cooked).
has_attr(1243746, any, bright).
has_attr(1243746, color, yellow).
has_attr(1243746, hposition, middle).
has_attr(1243746, vposition, middle).
has_rel(1243746, below, 1243751).
object(1243747).
has_attr(1243747, class, bacon).
has_attr(1243747, name, bacon).
has_attr(1243747, class, food).
has_attr(1243747, class, meat).
has_attr(1243747, color, dark).
has_attr(1243747, tone, dark).
has_attr(1243747, any, cooked).
has_attr(1243747, any, burnt).
has_attr(1243747, hposition, left).
has_attr(1243747, vposition, bottom).
has_attr(1243747, vposition, middle).
has_rel(1243747, to_the_left_of, 3827420).
object(1243745).
has_attr(1243745, class, sandwich).
has_attr(1243745, name, sandwich).
has_attr(1243745, class, food).
has_attr(1243745, color, brown).
has_attr(1243745, width, wide).
has_attr(1243745, size, large).
has_attr(1243745, any, seasoned).
has_attr(1243745, hposition, middle).
has_attr(1243745, vposition, middle).
has_rel(1243745, with, 1243748).
object(3827418).
has_attr(3827418, class, cheese).
has_attr(3827418, name, cheese).
has_attr(3827418, class, food).
has_attr(3827418, class, flavor).
has_attr(3827418, hposition, middle).
has_attr(3827418, vposition, middle).
object(3827419).
has_attr(3827419, class, person).
has_attr(3827419, name, person).
has_attr(3827419, class, person).
has_attr(3827419, hposition, right).
has_attr(3827419, vposition, bottom).
has_attr(3827419, vposition, middle).
has_rel(3827419, sitting_in, 3827420).
object(3827420).
has_attr(3827420, class, chair).
has_attr(3827420, name, chair).
has_attr(3827420, class, furniture).
has_attr(3827420, hposition, right).
has_attr(3827420, vposition, bottom).
has_attr(3827420, vposition, middle).
has_rel(3827420, to_the_right_of, 1243747).
object(1243748).
has_attr(1243748, class, meat).
has_attr(1243748, name, meat).
has_attr(1243748, class, food).
has_attr(1243748, class, meat).
has_attr(1243748, any, cooked).
has_attr(1243748, color, brown).
has_attr(1243748, hposition, middle).
has_attr(1243748, vposition, middle).
has_rel(1243748, inside, 1243745).
object(1243749).
has_attr(1243749, class, hand).
has_attr(1243749, name, hand).
has_attr(1243749, hposition, left).
has_attr(1243749, vposition, top).

% ------ question encoding ------
scene(0).
select(1, 0, person).
unique(2, 1).
query(3, 2, hposition).
end(3).
% ------ NL question ------

% On which side of the photo is the person?

% ------ Full answer ------

% The person is on the right of the image.
