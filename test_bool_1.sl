; Synthesize using only AND and NOT gates

(set-logic BV)

(define-fun customize ((a Bool) (b Bool)) Bool
  (or a b))

(synth-fun AIG ((a Bool) (b Bool)) Bool
           ((Start Bool ((and Start Start) (not Start) a b))))

(declare-var a Bool)
(declare-var b Bool)

(constraint
 (= (customize a b) 
    (AIG a b)
        ))


(check-synth)

; Solution:
;(define-fun AIG ((a Bool) (b Bool)) Bool
;(not (and (not a) (not b)))) 