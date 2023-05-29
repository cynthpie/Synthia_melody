class FrequencyModulator:
    """need to have smth like CEGEC CEGEC"""
    def __init__(self, notes=["C4", "E4", "G4"], note_lens=[0.5,0.5,0.5], duration=1.5, sample_rate=44100):
        self.notes = notes
        self.note_lens = note_lens
        self.duration = duration
        self._sample_rate = sample_rate
        self.counter = 0
        assert len(self.note_lens) == len(self.notes)
        length_flag = sum(self.note_lens)< self.duration
        while length_flag:
            for i in range(len(note_lens)):
                self.notes.append(notes[i])
                self.note_lens.append(note_lens[i])
                if sum(self.note_lens)> self.duration:
                    self.note_lens[-1] = max(self.note_lens[-1] - sum(self.note_lens) + self.duration, 0)
                    length_flag = False
                    break
        self.note_steps = [int(length*self._sample_rate) for length in self.note_lens]
        self.total_steps = int(self._sample_rate * self.duration)
        
        a = 0
        while sum(self.note_steps) != self.total_steps:   
                self.note_steps[-a] = max(self.note_steps[-a] - (sum(self.note_steps) - self.total_steps),0)
                a+=1
        assert sum(self.note_steps) == self.total_steps
        assert len(self.note_steps) == len(self.notes)
        assert all(step>=0 for step in self.note_steps)
        
    def get_freq(self): # list value you need
        self.val = 0
        try:
            self.note_steps[0]
        except IndexError:
            val = 0
            self.ended = True
            return val
        if self.counter <= self.note_steps[0]:
            assert len(self.note_steps) == len(self.notes)
            if self.counter < self.note_steps[0]:
                val = self.notes[0]
                self.counter+=1
            else:
                self.note_steps.pop(0)
                self.notes.pop(0)
                assert len(self.note_steps) == len(self.notes)
                try:
                    val = self.notes[0]
                except IndexError:
                    self.ended = True
                    val = 0
                self.counter = 0
            return val
        val = 0
        return val

    def __iter__(self): # get the value
        self.val = 0
        self.ended = False
        return self
    
    def __next__(self): # get the value
        self.val = self.get_freq()
        return self.val